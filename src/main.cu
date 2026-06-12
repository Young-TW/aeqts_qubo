#include <chrono>
#include <cstdio>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "kernels.cuh"
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
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    CUDA_CHECK(cudaSetDevice(local_rank % device_count));
#else
    const int world_rank = 0, world_size = 1;
#endif

    int iter = 1000;
    int n_items = 500;
    int N = 50;
    unsigned long long base_seed = 12345ULL;
    int run_id = 0;
    double P_penalty = 10.0;

    // ---- CLI args parse ----
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--seed" && i + 1 < argc) {
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

    curandStatePhilox4_32_10_t* d_states = nullptr;
    int total_measure_threads = N * n_items;

    CUDA_CHECK(
        cudaMalloc(&dQ, (size_t)n_items * (size_t)n_items * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(dQ, Qh.data(),
                          (size_t)n_items * (size_t)n_items * sizeof(double),
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_energy, (size_t)N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_idx, (size_t)N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_nei,
                          (size_t)N * (size_t)n_items * sizeof(unsigned char)));

    CUDA_CHECK(cudaMalloc(&d_alpha, (size_t)n_items * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_beta, (size_t)n_items * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_states, (size_t)total_measure_threads *
                                         sizeof(curandStatePhilox4_32_10_t)));

    // init RNG
    {
        int threads = 256;
        int blocks = (total_measure_threads + threads - 1) / threads;
        init_curand_states<<<blocks, threads>>>(d_states, actual_seed,
                                                total_measure_threads);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    double* d_global_best_energy = nullptr;
    unsigned char* d_global_best_sol = nullptr;
    CUDA_CHECK(cudaMalloc(&d_global_best_energy, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_global_best_sol,
                          (size_t)n_items * sizeof(unsigned char)));

    double* d_energy_sorted = nullptr;
    int* d_idx_sorted = nullptr;
    CUDA_CHECK(cudaMalloc(&d_energy_sorted, (size_t)N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_idx_sorted, (size_t)N * sizeof(int)));

    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
                                    d_energy, d_energy_sorted, d_idx,
                                    d_idx_sorted, N);

    CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));
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
    CUDA_CHECK(cudaMemcpy(d_alpha, alpha_h.data(),
                          (size_t)n_items * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_beta, beta_h.data(),
                          (size_t)n_items * sizeof(float),
                          cudaMemcpyHostToDevice));

    // Initialize GPU global best energy (use double)
    double init_energy_val = 1e30;
    CUDA_CHECK(cudaMemcpy(d_global_best_energy, &init_energy_val,
                          sizeof(double), cudaMemcpyHostToDevice));

    // ---- Initial evaluation ----
    {
        int threads = 256;
        int blocks = (total_measure_threads + threads - 1) / threads;
        generate_neighbours_kernel<<<blocks, threads>>>(d_alpha, d_beta, d_nei,
                                                        N, n_items, d_states);
        CUDA_CHECK(cudaGetLastError());

        qubo_energy_kernel_optimized<256>
            <<<N, 256>>>(d_nei, dQ, d_energy, n_items);
        CUDA_CHECK(cudaGetLastError());

        int seq_blocks = (N + threads - 1) / threads;
        init_sequence_kernel<<<seq_blocks, threads>>>(d_idx, N);

        cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
                                        d_energy, d_energy_sorted, d_idx,
                                        d_idx_sorted, N);

        update_global_best_kernel<<<1, 256>>>(d_energy_sorted, d_idx_sorted,
                                              d_nei, d_global_best_energy,
                                              d_global_best_sol, n_items);
    }

    // ---- Per-iteration timing ----
    CUDA_CHECK(cudaDeviceSynchronize());
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

        cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
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

    CUDA_CHECK(cudaDeviceSynchronize());
    auto t1 = std::chrono::high_resolution_clock::now();
    double total_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    double avg_ms = total_ms / (double)iter;
    double final_global_best_energy = 0.0;
    CUDA_CHECK(cudaMemcpy(&final_global_best_energy, d_global_best_energy,
                          sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(best_sol_h.data(), d_global_best_sol,
                          (size_t)n_items * sizeof(unsigned char),
                          cudaMemcpyDeviceToHost));

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
    CUDA_CHECK(cudaFree(d_states));
    CUDA_CHECK(cudaFree(d_beta));
    CUDA_CHECK(cudaFree(d_alpha));
    CUDA_CHECK(cudaFree(d_nei));
    CUDA_CHECK(cudaFree(d_idx));
    CUDA_CHECK(cudaFree(d_energy));
    CUDA_CHECK(cudaFree(dQ));
    CUDA_CHECK(cudaFree(d_global_best_energy));
    CUDA_CHECK(cudaFree(d_global_best_sol));
    CUDA_CHECK(cudaFree(d_temp_storage));
    CUDA_CHECK(cudaFree(d_idx_sorted));
    CUDA_CHECK(cudaFree(d_energy_sorted));

#ifdef USE_MPI
    MPI_Finalize();
#endif

    return 0;
}
