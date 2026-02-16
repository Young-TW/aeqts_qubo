#include "kernels.cuh"

#include <cstdio>
#include <vector>
#include <numeric>
#include <iostream>
#include <chrono>

// ------------------------------ Host: build QUBO matrix ------------------------------
// Same as Python build_teacher_qubo_matrix(values, weights, capacity, P)
static std::vector<float> build_teacher_qubo_matrix_host(
    const std::vector<float>& values,
    const std::vector<float>& weights,
    float capacity,
    float P)
{
    const int n = (int)values.size();
    std::vector<float> Q((size_t)n * (size_t)n, 0.0f);

    for (int i = 0; i < n; ++i) {
        float coeff_linear = -values[i] + 2.0f * P * (weights[i] * weights[i]) - 2.0f * P * capacity * weights[i];
        Q[(size_t)i * n + i] = coeff_linear;

        for (int j = i + 1; j < n; ++j) {
            float coeff_quad = 2.0f * P * weights[i] * weights[j];
            float v = coeff_quad * 0.5f;
            Q[(size_t)i * n + j] = v;
            Q[(size_t)j * n + i] = v;
        }
    }
    return Q;
}

// ------------------------------ Main ------------------------------
int main(int argc, char** argv)
{
    // ====== Match your classmate defaults ======
    const int experiment = 10;
    const int iter = 1000;
    const int n_items = 500;
    const int N = 50;

    // ---- seed parsing ----
    // Usage:
    //   aeqts_qubo_cuda.exe --seed 12345
    //   aeqts_qubo_cuda.exe 12345
    unsigned long long seed = 12345ULL;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--seed" && i + 1 < argc) {
            seed = std::stoull(argv[++i]);
        } else if (i == 1 && !a.empty() && a[0] != '-') {
            // allow first positional argument as seed
            seed = std::stoull(a);
        }
    }

    // weights: items = mod(range,10)+1 ; values=items+5 ; C = sum(weights)/2
    std::vector<float> weights(n_items), values(n_items);
    for (int i = 0; i < n_items; ++i) {
        float w = (float)((i % 10) + 1);
        weights[i] = w;
        values[i]  = w + 5.0f;
    }
    float sum_w = std::accumulate(weights.begin(), weights.end(), 0.0f);
    float C = sum_w / 2.0f;

    float P_penalty = 10.0f;

    std::cout << "Building QUBO matrix (Teacher formulation)...\n";
    std::vector<float> Qh = build_teacher_qubo_matrix_host(values, weights, C, P_penalty);

    // ====== Device allocations ======
    float *dQ = nullptr;
    float *d_energy = nullptr;
    int   *d_idx = nullptr;

    unsigned char* d_nei = nullptr; // N*n_items

    float *d_alpha = nullptr, *d_beta = nullptr;

    // For neighbour RNG: one state per (nbr,item) thread to match kernel indexing
    curandStatePhilox4_32_10_t* d_states = nullptr;
    int total_measure_threads = N * n_items;

    CUDA_CHECK(cudaMalloc(&dQ, (size_t)n_items * (size_t)n_items * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dQ, Qh.data(), (size_t)n_items * (size_t)n_items * sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_energy, (size_t)N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_idx, (size_t)N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_nei, (size_t)N * (size_t)n_items * sizeof(unsigned char)));

    CUDA_CHECK(cudaMalloc(&d_alpha, (size_t)n_items * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_beta,  (size_t)n_items * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_states, (size_t)total_measure_threads * sizeof(curandStatePhilox4_32_10_t)));

    // init RNG (match fixed seed behaviour)
    {
        int threads = 256;
        int blocks  = (total_measure_threads + threads - 1) / threads;
        init_curand_states<<<blocks, threads>>>(d_states, seed, total_measure_threads);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // 新增：在 Device 端配置存放「全局最佳解」的空間
    float *d_global_best_energy = nullptr;
    unsigned char *d_global_best_sol = nullptr;
    CUDA_CHECK(cudaMalloc(&d_global_best_energy, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_global_best_sol, (size_t)n_items * sizeof(unsigned char)));

    // 新增：CUB Radix Sort 屬於 Out-of-place 排序，需要準備輸出的陣列
    float *d_energy_sorted = nullptr;
    int   *d_idx_sorted = nullptr;
    CUDA_CHECK(cudaMalloc(&d_energy_sorted, (size_t)N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_idx_sorted, (size_t)N * sizeof(int)));

    // 新增：向 CUB 查詢排序 N 個元素需要多少暫存記憶體 (temp_storage)
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
                                    d_energy, d_energy_sorted,
                                    d_idx, d_idx_sorted, N);
    
    // 預先配置好暫存記憶體，這樣迴圈內就不會有任何 cudaMalloc 開銷了
    CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));

    // host buffers for reporting
    std::vector<unsigned char> best_sol_h(n_items);

    std::cout << "\n=======================================\n";
    std::cout << "Start experiments\n";
    std::cout << "Seed=" << seed << "\n";
    std::cout << "P=" << P_penalty << ", Items=" << n_items << ", Capacity=" << C << "\n";
    std::cout << "N=" << N << ", Iter=" << iter << "\n";
    std::cout << "=======================================\n\n";

    for (int e = 0; e < experiment; ++e) {

        // init qindividuals to 1/sqrt(2)
        float init = 1.0f / std::sqrt(2.0f);
        std::vector<float> alpha_h(n_items, init), beta_h(n_items, init);
        CUDA_CHECK(cudaMemcpy(d_alpha, alpha_h.data(), (size_t)n_items * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_beta,  beta_h.data(),  (size_t)n_items * sizeof(float), cudaMemcpyHostToDevice));

        // 初始化 GPU 端的全域最佳能量為極大值
        float init_energy_val = 1e30f;
        CUDA_CHECK(cudaMemcpy(d_global_best_energy, &init_energy_val, sizeof(float), cudaMemcpyHostToDevice));

        // ---- Initial evaluation ----
        {
            // 1) generate neighbours once
            int threads = 256;
            int blocks  = (total_measure_threads + threads - 1) / threads;
            generate_neighbours_kernel<<<blocks, threads>>>(d_alpha, d_beta, d_nei, N, n_items, d_states);
            CUDA_CHECK(cudaGetLastError());

            // 2) energies (Optimized)
            {
                int threads = 256; // 對應上面的 BLOCK_THREADS
                int blocks  = N;   // N=50，發射 50 個 Blocks
                qubo_energy_kernel_optimized<256><<<blocks, threads>>>(d_nei, dQ, d_energy, n_items);
                CUDA_CHECK(cudaGetLastError());
            }

            // 3) reset indices every iteration (取代 thrust::sequence)
            {
                int seq_threads = 256;
                int seq_blocks = (N + seq_threads - 1) / seq_threads;
                init_sequence_kernel<<<seq_blocks, seq_threads>>>(d_idx, N);
            }

            // 4) sort by energy ascending (取代 thrust::sort_by_key)
            {
                cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
                                                d_energy, d_energy_sorted,
                                                d_idx, d_idx_sorted, N);
            }

            // 5) update global best (注意這裡傳入的是 d_energy_sorted 與 d_idx_sorted)
            {
                int update_threads = 256;
                update_global_best_kernel<<<1, update_threads>>>(d_energy_sorted, d_idx_sorted, d_nei, d_global_best_energy, d_global_best_sol, n_items);
            }
        }

        // ---- Per-iteration timing ----
        // 修正：計時器包在整個迴圈外，不計算 Kernel 啟動與 Host 端迴圈的微小開銷，更能反映真實的 GPU 執行時間
        CUDA_CHECK(cudaDeviceSynchronize()); // 確保初始化的工作全數完成
        auto t0 = std::chrono::high_resolution_clock::now();

        for (int it = 0; it < iter; ++it) {
            // 1) generate neighbours
            {
                int threads = 256;
                int blocks  = (total_measure_threads + threads - 1) / threads;
                generate_neighbours_kernel<<<blocks, threads>>>(d_alpha, d_beta, d_nei, N, n_items, d_states);
                CUDA_CHECK(cudaGetLastError());
            }

            // 2) energies (Optimized)
            {
                int threads = 256; // 對應上面的 BLOCK_THREADS
                int blocks  = N;   // N=50，發射 50 個 Blocks
                qubo_energy_kernel_optimized<256><<<blocks, threads>>>(d_nei, dQ, d_energy, n_items);
                CUDA_CHECK(cudaGetLastError());
            }

            // 3) reset indices every iteration (取代 thrust::sequence)
            {
                int seq_threads = 256;
                int seq_blocks = (N + seq_threads - 1) / seq_threads;
                init_sequence_kernel<<<seq_blocks, seq_threads>>>(d_idx, N);
            }

            // 4) sort by energy ascending (取代 thrust::sort_by_key)
            {
                cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
                                                d_energy, d_energy_sorted,
                                                d_idx, d_idx_sorted, N);
            }

            // 5) update global best (注意這裡傳入的是 d_energy_sorted 與 d_idx_sorted)
            {
                int update_threads = 256;
                update_global_best_kernel<<<1, update_threads>>>(d_energy_sorted, d_idx_sorted, d_nei, d_global_best_energy, d_global_best_sol, n_items);
            }

            // 6) updateQ uses sorted best/worst pairing (注意這裡傳入的是 d_idx_sorted)
            {
                int threads = 128;
                int blocks  = (n_items + threads - 1) / threads;
                updateQ_kernel<<<blocks, threads>>>(d_nei, d_idx_sorted, d_alpha, d_beta, N, n_items);
            }
        }

        // 確保 1000 次迴圈的所有 Kernel 都執行完畢
        CUDA_CHECK(cudaDeviceSynchronize());
        auto t1 = std::chrono::high_resolution_clock::now();
        double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double avg_ms = total_ms / (double)iter;

        // 迴圈結束後，將結果抓回 CPU 進行驗證
        float final_global_best_energy = 0.0f;
        CUDA_CHECK(cudaMemcpy(&final_global_best_energy, d_global_best_energy, sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(best_sol_h.data(), d_global_best_sol, (size_t)n_items * sizeof(unsigned char), cudaMemcpyDeviceToHost));

        // compute real value & weight on host
        float final_w = 0.0f, final_v = 0.0f;
        for (int i = 0; i < n_items; ++i) {
            if (best_sol_h[i]) {
                final_w += weights[i];
                final_v += values[i];
            }
        }

        bool valid = (final_w <= C + 1e-5f);
        std::cout << "Run " << (e + 1)
                  << ": Energy=" << final_global_best_energy
                  << " | Val=" << final_v
                  << " | W=" << final_w << "/" << C
                  << " | " << (valid ? "VALID" : "OVERWEIGHT")
                  << " | AvgIter=" << avg_ms << " ms\n";
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

    return 0;
}
