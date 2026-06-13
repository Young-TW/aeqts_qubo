#include <cmath>
#include <cstdio>
#include <iostream>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "config.h"
#include "qubo_matrix.h"
#include "solver.h"

#ifdef USE_MPI
#include <mpi.h>
#endif

// ------------------------------ Main ------------------------------
// 純 C++ 入口:負責參數解析、問題建構、QUBO 矩陣建立、MPI 島嶼彙整與輸出。
// 所有 GPU 運算都委派給 solver.h 介面(由 CUDA/HIP 後端實作)。
int main(int argc, char** argv) {
#ifdef USE_MPI
    MPI_Init(&argc, &argv);
    int world_rank = 0, world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // 取得 node 內的 local rank,用來綁定各自的 GPU
    MPI_Comm node_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, world_rank,
                        MPI_INFO_NULL, &node_comm);
    int local_rank = 0;
    MPI_Comm_rank(node_comm, &local_rank);
    MPI_Comm_free(&node_comm);
    gpu_set_device(local_rank);
#else
    const int world_rank = 0, world_size = 1;
    gpu_set_device(0);
#endif

    // ---- 先決定設定檔路徑(可由 --config 覆寫,預設 config/case.conf) ----
    std::string config_path = "config/case.conf";
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) {
            config_path = argv[++i];
        }
    }

    // ---- 從設定檔載入參數 ----
    Config cfg;
    if (!load_config(config_path, cfg)) {
        if (world_rank == 0) {
            std::cerr << "[config] 無法開啟 '" << config_path
                      << "',改用內建預設值。\n";
        }
    } else if (world_rank == 0) {
        std::cout << "Loaded config from '" << config_path << "'\n";
    }

    int iter = cfg.iter;
    int n_items = cfg.n_items;
    int N = cfg.N;
    unsigned long long base_seed = cfg.base_seed;
    int run_id = cfg.run_id;
    float P_penalty = (float)cfg.P_penalty;

    // ---- CLI args parse(覆寫設定檔的值) ----
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) {
            ++i;  // 已於上方處理
        } else if (arg == "--seed" && i + 1 < argc) {
            base_seed = std::stoull(argv[++i]);
        } else if (arg == "--items" && i + 1 < argc) {
            n_items = std::stoi(argv[++i]);
        } else if (arg == "--N" && i + 1 < argc) {
            N = std::stoi(argv[++i]);
        } else if (arg == "--iter" && i + 1 < argc) {
            iter = std::stoi(argv[++i]);
        } else if (arg == "--run_id" && i + 1 < argc) {
            run_id = std::stoi(argv[++i]);
        } else if ((arg == "-P" || arg == "--penalty") && i + 1 < argc) {
            P_penalty = std::stof(argv[++i]);
        } else if (i == 1 && !arg.empty() && arg[0] != '-') {
            base_seed = std::stoull(arg);
        }
    }

    // 島嶼模型:每個 rank 以不同種子做獨立搜尋。
    // run_id (Slurm array) 與 world_rank 都納入,避免跨任務 / 跨 rank 撞種子。
    unsigned long long actual_seed =
        base_seed +
        (unsigned long long)run_id * (unsigned long long)world_size +
        (unsigned long long)world_rank;

    std::vector<float> weights(n_items), values(n_items);
    for (int i = 0; i < n_items; ++i) {
        float w = (float)((i % 10) + 1);
        weights[i] = w;
        values[i] = w + 5.0f;
    }
    float sum_w = std::accumulate(weights.begin(), weights.end(), 0.0f);
    float C = sum_w / 2.0f;

    if (world_rank == 0) {
        std::cout << "Building QUBO matrix (Teacher formulation)...\n";
    }
    std::vector<float> Qh =
        build_teacher_qubo_matrix_host(values, weights, C, P_penalty);

    if (world_rank == 0) {
        std::cout << "\n=======================================\n";
        std::cout << "Start experiment (Run ID: " << run_id << ")\n";
        std::cout << "MPI ranks (islands): " << world_size << "\n";
        std::cout << "Seed=" << actual_seed << " (Base=" << base_seed << ")\n";
        std::cout << "P=" << P_penalty << ", Items=" << n_items
                  << ", Capacity=" << C << "\n";
        std::cout << "N=" << N << ", Iter=" << iter << "\n";
        std::cout << "=======================================\n\n";
    }

    // ====== 委派 GPU 搜尋給後端 ======
    AeqtsParams params{iter, n_items, N, actual_seed};
    AeqtsResult result = run_aeqts(params, Qh);

    float final_global_best_energy = result.best_energy;
    std::vector<unsigned char> best_sol_h = std::move(result.best_solution);
    double avg_ms = result.avg_iter_ms;  // 計時保留 double

    int best_rank = world_rank;
#ifdef USE_MPI
    // ---- 島嶼模型彙整:跨 rank 挑出能量最低者,並廣播其解 ----
    struct {
        float energy;
        int rank;
    } local_pair, global_pair;
    local_pair.energy = final_global_best_energy;
    local_pair.rank = world_rank;
    MPI_Allreduce(&local_pair, &global_pair, 1, MPI_FLOAT_INT, MPI_MINLOC,
                  MPI_COMM_WORLD);
    final_global_best_energy = global_pair.energy;
    best_rank = global_pair.rank;
    // 由勝出的 rank 把最佳解廣播給所有人
    MPI_Bcast(best_sol_h.data(), n_items, MPI_UNSIGNED_CHAR, best_rank,
              MPI_COMM_WORLD);
#endif

    float final_w = 0.0f, final_v = 0.0f;
    for (int i = 0; i < n_items; ++i) {
        if (best_sol_h[i]) {
            final_w += weights[i];
            final_v += values[i];
        }
    }

    bool valid = (final_w <= C + 1e-5f);
    if (world_rank == 0) {
        std::cout << ": Energy=" << final_global_best_energy
                  << " | Val=" << final_v << " | W=" << final_w << "/" << C
                  << " | " << (valid ? "VALID" : "OVERWEIGHT")
                  << " | BestRank=" << best_rank << " | AvgIter=" << avg_ms
                  << " ms\n";
    }

#ifdef USE_MPI
    MPI_Finalize();
#endif

    return 0;
}
