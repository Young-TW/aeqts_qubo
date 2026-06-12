#include <chrono>
#include <cmath>
#include <vector>

#include "kernels.cuh"
#include "solver.h"

void gpu_set_device(int local_rank) {
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count > 0) {
        CUDA_CHECK(cudaSetDevice(local_rank % device_count));
    }
}

AeqtsResult run_aeqts(const AeqtsParams& params, const std::vector<double>& Qh) {
    const int iter = params.iter;
    const int n_items = params.n_items;
    const int N = params.N;
    const unsigned long long actual_seed = params.seed;

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

    AeqtsResult result;
    result.best_solution.resize(n_items);
    result.avg_iter_ms = total_ms / (double)iter;

    CUDA_CHECK(cudaMemcpy(&result.best_energy, d_global_best_energy,
                          sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(result.best_solution.data(), d_global_best_sol,
                          (size_t)n_items * sizeof(unsigned char),
                          cudaMemcpyDeviceToHost));

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

    return result;
}
