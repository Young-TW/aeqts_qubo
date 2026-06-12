#include <chrono>
#include <cstdio>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "config.h"
#include "kernels.hpp"
#include "qubo_matrix.h"

#ifdef USE_MPI
#include <mpi.h>
#endif

// ------------------------------ Main ------------------------------
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

    int device_count = 0;
    CUDA_CHECK(hipGetDeviceCount(&device_count));
    CUDA_CHECK(hipSetDevice(local_rank % device_count));
#else
    const int world_rank = 0, world_size = 1;
#endif

    // ---- 先決定設定檔路徑(可由 --config 覆寫,預設 config/case.toml) ----
    std::string config_path = "config/case.toml";
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
    double P_penalty = cfg.P_penalty;

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
            P_penalty = std::stod(argv[++i]);
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

    std::vector<double> weights(n_items), values(n_items);
    for (int i = 0; i < n_items; ++i) {
        double w = (double)((i % 10) + 1);
        weights[i] = w;
        values[i] = w + 5.0;
    }
    double sum_w = std::accumulate(weights.begin(), weights.end(), 0.0);
    double C = sum_w / 2.0;

    if (world_rank == 0) {
        std::cout << "Building QUBO matrix (Teacher formulation)...\n";
    }
    std::vector<double> Qh =
        build_teacher_qubo_matrix_host(values, weights, C, P_penalty);

    // ====== Device allocations ======
    double* dQ = nullptr;
    double* d_energy = nullptr;
    int* d_idx = nullptr;
    unsigned char* d_nei = nullptr;
    float *d_alpha = nullptr, *d_beta = nullptr;

    hiprandStatePhilox4_32_10_t* d_states = nullptr;
    int total_measure_threads = N * n_items;

    CUDA_CHECK(
        hipMalloc(&dQ, (size_t)n_items * (size_t)n_items * sizeof(double)));
    CUDA_CHECK(hipMemcpy(dQ, Qh.data(),
                          (size_t)n_items * (size_t)n_items * sizeof(double),
                          hipMemcpyHostToDevice));

    CUDA_CHECK(hipMalloc(&d_energy, (size_t)N * sizeof(double)));
    CUDA_CHECK(hipMalloc(&d_idx, (size_t)N * sizeof(int)));
    CUDA_CHECK(hipMalloc(&d_nei,
                          (size_t)N * (size_t)n_items * sizeof(unsigned char)));

    CUDA_CHECK(hipMalloc(&d_alpha, (size_t)n_items * sizeof(float)));
    CUDA_CHECK(hipMalloc(&d_beta, (size_t)n_items * sizeof(float)));

    CUDA_CHECK(hipMalloc(&d_states, (size_t)total_measure_threads *
                                         sizeof(hiprandStatePhilox4_32_10_t)));

    // init RNG
    {
        int threads = 256;
        int blocks = (total_measure_threads + threads - 1) / threads;
        init_curand_states<<<blocks, threads>>>(d_states, actual_seed,
                                                total_measure_threads);
        CUDA_CHECK(hipGetLastError());
        CUDA_CHECK(hipDeviceSynchronize());
    }

    double* d_global_best_energy = nullptr;
    unsigned char* d_global_best_sol = nullptr;
    CUDA_CHECK(hipMalloc(&d_global_best_energy, sizeof(double)));
    CUDA_CHECK(hipMalloc(&d_global_best_sol,
                          (size_t)n_items * sizeof(unsigned char)));

    double* d_energy_sorted = nullptr;
    int* d_idx_sorted = nullptr;
    CUDA_CHECK(hipMalloc(&d_energy_sorted, (size_t)N * sizeof(double)));
    CUDA_CHECK(hipMalloc(&d_idx_sorted, (size_t)N * sizeof(int)));

    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    hipcub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
                                    d_energy, d_energy_sorted, d_idx,
                                    d_idx_sorted, N);

    CUDA_CHECK(hipMalloc(&d_temp_storage, temp_storage_bytes));
    std::vector<unsigned char> best_sol_h(n_items);

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

    float init = 1.0f / std::sqrt(2.0f);
    std::vector<float> alpha_h(n_items, init), beta_h(n_items, init);
    CUDA_CHECK(hipMemcpy(d_alpha, alpha_h.data(),
                          (size_t)n_items * sizeof(float),
                          hipMemcpyHostToDevice));
    CUDA_CHECK(hipMemcpy(d_beta, beta_h.data(),
                          (size_t)n_items * sizeof(float),
                          hipMemcpyHostToDevice));

    // Initialize GPU global best energy (use double)
    double init_energy_val = 1e30;
    CUDA_CHECK(hipMemcpy(d_global_best_energy, &init_energy_val,
                          sizeof(double), hipMemcpyHostToDevice));

    // ---- Initial evaluation ----
    {
        int threads = 256;
        int blocks = (total_measure_threads + threads - 1) / threads;
        generate_neighbours_kernel<<<blocks, threads>>>(d_alpha, d_beta, d_nei,
                                                        N, n_items, d_states);
        CUDA_CHECK(hipGetLastError());

        qubo_energy_kernel_optimized<256>
            <<<N, 256>>>(d_nei, dQ, d_energy, n_items);
        CUDA_CHECK(hipGetLastError());

        int seq_blocks = (N + threads - 1) / threads;
        init_sequence_kernel<<<seq_blocks, threads>>>(d_idx, N);

        hipcub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
                                        d_energy, d_energy_sorted, d_idx,
                                        d_idx_sorted, N);

        update_global_best_kernel<<<1, 256>>>(d_energy_sorted, d_idx_sorted,
                                              d_nei, d_global_best_energy,
                                              d_global_best_sol, n_items);
    }

    // ---- Per-iteration timing ----
    CUDA_CHECK(hipDeviceSynchronize());
    auto t0 = std::chrono::high_resolution_clock::now();

    for (int it = 0; it < iter; ++it) {
        int threads = 256;
        int blocks = (total_measure_threads + threads - 1) / threads;
        generate_neighbours_kernel<<<blocks, threads>>>(d_alpha, d_beta, d_nei,
                                                        N, n_items, d_states);

        qubo_energy_kernel_optimized<256>
            <<<N, 256>>>(d_nei, dQ, d_energy, n_items);

        int seq_blocks = (N + threads - 1) / threads;
        init_sequence_kernel<<<seq_blocks, threads>>>(d_idx, N);

        hipcub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
                                        d_energy, d_energy_sorted, d_idx,
                                        d_idx_sorted, N);

        update_global_best_kernel<<<1, 256>>>(d_energy_sorted, d_idx_sorted,
                                              d_nei, d_global_best_energy,
                                              d_global_best_sol, n_items);

        int update_threads = 128;
        int update_blocks = (n_items + update_threads - 1) / update_threads;
        updateQ_kernel<<<update_blocks, update_threads>>>(
            d_nei, d_idx_sorted, d_alpha, d_beta, N, n_items);
    }

    CUDA_CHECK(hipDeviceSynchronize());
    auto t1 = std::chrono::high_resolution_clock::now();
    double total_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    double avg_ms = total_ms / (double)iter;
    double final_global_best_energy = 0.0;
    CUDA_CHECK(hipMemcpy(&final_global_best_energy, d_global_best_energy,
                          sizeof(double), hipMemcpyDeviceToHost));
    CUDA_CHECK(hipMemcpy(best_sol_h.data(), d_global_best_sol,
                          (size_t)n_items * sizeof(unsigned char),
                          hipMemcpyDeviceToHost));

    int best_rank = world_rank;
#ifdef USE_MPI
    // ---- 島嶼模型彙整:跨 rank 挑出能量最低者,並廣播其解 ----
    struct {
        double energy;
        int rank;
    } local_pair, global_pair;
    local_pair.energy = final_global_best_energy;
    local_pair.rank = world_rank;
    MPI_Allreduce(&local_pair, &global_pair, 1, MPI_DOUBLE_INT, MPI_MINLOC,
                  MPI_COMM_WORLD);
    final_global_best_energy = global_pair.energy;
    best_rank = global_pair.rank;
    // 由勝出的 rank 把最佳解廣播給所有人
    MPI_Bcast(best_sol_h.data(), n_items, MPI_UNSIGNED_CHAR, best_rank,
              MPI_COMM_WORLD);
#endif

    double final_w = 0.0, final_v = 0.0;
    for (int i = 0; i < n_items; ++i) {
        if (best_sol_h[i]) {
            final_w += weights[i];
            final_v += values[i];
        }
    }

    bool valid = (final_w <= C + 1e-5);
    if (world_rank == 0) {
        std::cout << ": Energy=" << final_global_best_energy
                  << " | Val=" << final_v << " | W=" << final_w << "/" << C
                  << " | " << (valid ? "VALID" : "OVERWEIGHT")
                  << " | BestRank=" << best_rank << " | AvgIter=" << avg_ms
                  << " ms\n";
    }

    // cleanup
    CUDA_CHECK(hipFree(d_states));
    CUDA_CHECK(hipFree(d_beta));
    CUDA_CHECK(hipFree(d_alpha));
    CUDA_CHECK(hipFree(d_nei));
    CUDA_CHECK(hipFree(d_idx));
    CUDA_CHECK(hipFree(d_energy));
    CUDA_CHECK(hipFree(dQ));
    CUDA_CHECK(hipFree(d_global_best_energy));
    CUDA_CHECK(hipFree(d_global_best_sol));
    CUDA_CHECK(hipFree(d_temp_storage));
    CUDA_CHECK(hipFree(d_idx_sorted));
    CUDA_CHECK(hipFree(d_energy_sorted));

#ifdef USE_MPI
    MPI_Finalize();
#endif

    return 0;
}
