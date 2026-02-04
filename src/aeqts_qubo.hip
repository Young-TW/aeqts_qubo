#include "hip/hip_runtime.h"
// aeqts_qubo_cuda.cu
// AEQTS + QUBO (Teacher formulation, no slack, fixed P) : CUDA parallel version
// - Parallel: neighbour generation (measure), QUBO energy eval, updateQ per item
// - Sort: thrust::sort_by_key energies (ascending, smaller is better)

#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
// #define _USE_MATH_DEFINES
// #include <cmath>


#include <cmath>
constexpr float PI_F = 3.14159265358979323846f;

#include <cstdio>
#include <vector>
#include <numeric>
#include <iostream>
#include <chrono>

#define CUDA_CHECK(call) do {                                  \
  hipError_t _e = (call);                                     \
  if (_e != hipSuccess) {                                     \
    std::fprintf(stderr, "CUDA error %s:%d: %s\n",             \
      __FILE__, __LINE__, hipGetErrorString(_e));             \
    std::exit(1);                                              \
  }                                                            \
} while(0)

// static inline float sqr(float x) { return x * x; }

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
            // Python does: Q[i,j]=coeff/2 and Q[j,i]=coeff/2 (sym)
            float v = coeff_quad * 0.5f;
            Q[(size_t)i * n + j] = v;
            Q[(size_t)j * n + i] = v;
        }
    }
    return Q;
}

// ------------------------------ Device: RNG init ------------------------------
__global__ void init_curand_states(hiprandStatePhilox4_32_10_t* states, unsigned long long seed, int total_threads)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_threads) return;
    hiprand_init(seed, tid, 0, &states[tid]);
}

// ------------------------------ Device: generate neighbours (measure) ------------------------------
// qindividuals: n_items x 2 (alpha, beta). Probability of 1 is beta^2, same as Python:
// rand > probs ? 0 : 1
__global__ void generate_neighbours_kernel(
    const float* __restrict__ q_alpha,
    const float* __restrict__ q_beta,
    unsigned char* __restrict__ neighbours, // N x n_items (0/1)
    int N,
    int n_items,
    hiprandStatePhilox4_32_10_t* states)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // 0 .. N*n_items-1
    int total = N * n_items;
    if (idx >= total) return;

    int nbr = idx / n_items;
    int i   = idx - nbr * n_items;

    // one RNG state per thread index
    hiprandStatePhilox4_32_10_t st = states[idx];
    float r = hiprand_uniform(&st); // (0,1]
    states[idx] = st;

    float p1 = q_beta[i] * q_beta[i];
    neighbours[(size_t)nbr * n_items + i] = (r > p1) ? 0 : 1;
}

// ------------------------------ Device: QUBO energy eval ------------------------------
// Energy = x^T Q x  (x is 0/1)
// For symmetric Q, full formula is fine.
__global__ void qubo_energy_kernel(
    const unsigned char* __restrict__ neighbours, // N x n_items
    const float* __restrict__ Q,                  // n_items x n_items
    float* __restrict__ energies,                 // N
    int N,
    int n_items)
{
    int nbr = blockIdx.x * blockDim.x + threadIdx.x;
    if (nbr >= N) return;

    const unsigned char* x = neighbours + (size_t)nbr * n_items;

    float e = 0.0f;
    // x^T Q x
    for (int i = 0; i < n_items; ++i) {
        if (!x[i]) continue;
        const float* Qi = Q + (size_t)i * n_items;
        // sum over j
        for (int j = 0; j < n_items; ++j) {
            if (x[j]) e += Qi[j];
        }
    }
    energies[nbr] = e;
}

// ------------------------------ Device: updateQ (per item) ------------------------------
// Python:
// best_group = neighbours[:num_pairs]
// worst_group = neighbours[N-1 : N-num_pairs-1 : -1]
// diff_matrix = best_group - worst_group
// t = 1..num_pairs, base_thetas=(0.01*pi)/t
// raw_angles = sum(diff_matrix * base_thetas, axis=0)
// sign_mask = (alpha*beta<0) ? -1 : 1
// rotate (alpha,beta) by final_theta
__global__ void updateQ_kernel(
    const unsigned char* __restrict__ neighbours, // N x n_items
    const int* __restrict__ sorted_idx,           // N (ascending energies)
    float* __restrict__ q_alpha,
    float* __restrict__ q_beta,
    int N,
    int n_items)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_items) return;

    int num_pairs = N / 2;
    float raw_angle = 0.0f;

    for (int t = 0; t < num_pairs; ++t) {
        int best_nbr  = sorted_idx[t];
        int worst_nbr = sorted_idx[N - 1 - t];

        unsigned char xb = neighbours[(size_t)best_nbr  * n_items + i];
        unsigned char xw = neighbours[(size_t)worst_nbr * n_items + i];

        int diff = (int)xb - (int)xw; // -1,0,1
        float base_theta = (0.01f * PI_F) / (float)(t + 1);
        raw_angle += (float)diff * base_theta;
    }

    float a = q_alpha[i];
    float b = q_beta[i];
    float sign = (a * b < 0.0f) ? -1.0f : 1.0f;
    float theta = raw_angle * sign;

    float c = cosf(theta);
    float s = sinf(theta);

    float new_a = a * c - b * s;
    float new_b = a * s + b * c;

    q_alpha[i] = new_a;
    q_beta[i]  = new_b;
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
    hiprandStatePhilox4_32_10_t* d_states = nullptr;
    int total_measure_threads = N * n_items;

    CUDA_CHECK(hipMalloc(&dQ, (size_t)n_items * (size_t)n_items * sizeof(float)));
    CUDA_CHECK(hipMemcpy(dQ, Qh.data(), (size_t)n_items * (size_t)n_items * sizeof(float), hipMemcpyHostToDevice));

    CUDA_CHECK(hipMalloc(&d_energy, (size_t)N * sizeof(float)));
    CUDA_CHECK(hipMalloc(&d_idx, (size_t)N * sizeof(int)));
    CUDA_CHECK(hipMalloc(&d_nei, (size_t)N * (size_t)n_items * sizeof(unsigned char)));

    CUDA_CHECK(hipMalloc(&d_alpha, (size_t)n_items * sizeof(float)));
    CUDA_CHECK(hipMalloc(&d_beta,  (size_t)n_items * sizeof(float)));

    CUDA_CHECK(hipMalloc(&d_states, (size_t)total_measure_threads * sizeof(hiprandStatePhilox4_32_10_t)));

    // init RNG (match fixed seed behaviour)
    {
        int threads = 256;
        int blocks  = (total_measure_threads + threads - 1) / threads;
        init_curand_states<<<blocks, threads>>>(d_states, seed, total_measure_threads);
        CUDA_CHECK(hipGetLastError());
        CUDA_CHECK(hipDeviceSynchronize());
    }

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
        CUDA_CHECK(hipMemcpy(d_alpha, alpha_h.data(), (size_t)n_items * sizeof(float), hipMemcpyHostToDevice));
        CUDA_CHECK(hipMemcpy(d_beta,  beta_h.data(),  (size_t)n_items * sizeof(float), hipMemcpyHostToDevice));

        float global_best_energy = 1e30f;

        // ---- Initial evaluation (match Python: init_nbrs + find_best_worst) ----
        {
            // 1) generate neighbours once
            int threads = 256;
            int blocks  = (total_measure_threads + threads - 1) / threads;
            generate_neighbours_kernel<<<blocks, threads>>>(d_alpha, d_beta, d_nei, N, n_items, d_states);
            CUDA_CHECK(hipGetLastError());

            // 2) energies
            int t2 = 128;
            int b2 = (N + t2 - 1) / t2;
            qubo_energy_kernel<<<b2, t2>>>(d_nei, dQ, d_energy, N, n_items);
            CUDA_CHECK(hipGetLastError());

            // 3) reset indices (match Python argsort)
            {
                thrust::device_ptr<int> ip(d_idx);
                thrust::sequence(ip, ip + N);
            }

            // 4) sort by energy ascending
            {
                thrust::device_ptr<float> ep(d_energy);
                thrust::device_ptr<int>   ip(d_idx);
                thrust::sort_by_key(ep, ep + N, ip);
            }

            // 5) set global best from the best in sorted list
            CUDA_CHECK(hipMemcpy(&global_best_energy, d_energy, sizeof(float), hipMemcpyDeviceToHost));

            int best_idx = 0;
            CUDA_CHECK(hipMemcpy(&best_idx, d_idx, sizeof(int), hipMemcpyDeviceToHost));
            CUDA_CHECK(hipMemcpy(best_sol_h.data(),
                                  d_nei + (size_t)best_idx * (size_t)n_items,
                                  (size_t)n_items * sizeof(unsigned char),
                                  hipMemcpyDeviceToHost));
        }

        // ---- Per-iteration timing ----
        double iter_ms_sum = 0.0;

        double T = 0.0;

        for (int it = 0; it < iter; ++it) {
            auto t0 = std::chrono::high_resolution_clock::now();

            // 1) generate neighbours
            {
                int threads = 256;
                int blocks  = (total_measure_threads + threads - 1) / threads;
                generate_neighbours_kernel<<<blocks, threads>>>(d_alpha, d_beta, d_nei, N, n_items, d_states);
                CUDA_CHECK(hipGetLastError());
            }

            // 2) energies
            {
                int threads = 128;
                int blocks  = (N + threads - 1) / threads;
                qubo_energy_kernel<<<blocks, threads>>>(d_nei, dQ, d_energy, N, n_items);
                CUDA_CHECK(hipGetLastError());
            }

            // 3) reset indices every iteration (match Python argsort)
            {
                thrust::device_ptr<int> ip(d_idx);
                thrust::sequence(ip, ip + N);
            }

            // 4) sort by energy ascending
            {
                thrust::device_ptr<float> ep(d_energy);
                thrust::device_ptr<int>   ip(d_idx);
                thrust::sort_by_key(ep, ep + N, ip);
            }

            // 5) update global best (compare current best)
            float current_best_energy = 0.0f;
            CUDA_CHECK(hipMemcpy(&current_best_energy, d_energy, sizeof(float), hipMemcpyDeviceToHost));

            if (current_best_energy < global_best_energy) {
                global_best_energy = current_best_energy;

                int best_idx = 0;
                CUDA_CHECK(hipMemcpy(&best_idx, d_idx, sizeof(int), hipMemcpyDeviceToHost));
                CUDA_CHECK(hipMemcpy(best_sol_h.data(),
                                      d_nei + (size_t)best_idx * (size_t)n_items,
                                      (size_t)n_items * sizeof(unsigned char),
                                      hipMemcpyDeviceToHost));
            }

            // 6) updateQ uses sorted best/worst pairing
            {
                int threads = 128;
                int blocks  = (n_items + threads - 1) / threads;
                updateQ_kernel<<<blocks, threads>>>(d_nei, d_idx, d_alpha, d_beta, N, n_items);
                CUDA_CHECK(hipGetLastError());
            }

            // Make timing meaningful for GPU work
            CUDA_CHECK(hipDeviceSynchronize());

            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            iter_ms_sum += ms;
            T += ms;

            
        }
        // std::cout << "  Run " << (e + 1) << ": " << T << " ms\n";

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
                  << ": Energy=" << global_best_energy
                  << " | Val=" << final_v
                  << " | W=" << final_w << "/" << C
                  << " | " << (valid ? "VALID" : "OVERWEIGHT")
                  << " | AvgIter=" << (iter_ms_sum / (double)iter) << " ms\n";
                //   << T/1000.0 << " s\n";
    }

    // cleanup
    CUDA_CHECK(hipFree(d_states));
    CUDA_CHECK(hipFree(d_beta));
    CUDA_CHECK(hipFree(d_alpha));
    CUDA_CHECK(hipFree(d_nei));
    CUDA_CHECK(hipFree(d_idx));
    CUDA_CHECK(hipFree(d_energy));
    CUDA_CHECK(hipFree(dQ));

    return 0;
}
