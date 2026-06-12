#pragma once

#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

#include <hipcub/hipcub.hpp>

constexpr float PI_F = 3.14159265358979323846f;

// ------------------------------ 巨集 ------------------------------
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        hipError_t _e = (call);                                               \
        if (_e != hipSuccess) {                                               \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                         hipGetErrorString(_e));                              \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

// ------------------------------ Kernel 宣告 ------------------------------

__global__ void init_sequence_kernel(int* arr, int n);

__global__ void init_curand_states(hiprandStatePhilox4_32_10_t* states,
                                   unsigned long long seed, int total_threads);

__global__ void generate_neighbours_kernel(
    const float* __restrict__ q_alpha, const float* __restrict__ q_beta,
    unsigned char* __restrict__ neighbours,  // N x n_items (0/1)
    int N, int n_items, hiprandStatePhilox4_32_10_t* states);

__global__ void updateQ_kernel(
    const unsigned char* __restrict__ neighbours,  // N x n_items
    const int* __restrict__ sorted_idx,            // N (ascending energies)
    float* __restrict__ q_alpha, float* __restrict__ q_beta, int N,
    int n_items);

__global__ void update_global_best_kernel(
    const double* __restrict__ sorted_energies,
    const int* __restrict__ sorted_idx,
    const unsigned char* __restrict__ neighbours,
    double* __restrict__ global_best_energy,
    unsigned char* __restrict__ global_best_sol, int n_items);

// ------------------------------ Template Kernel 實作
// ------------------------------ 注意：樣板函式必須實作在標頭檔中
template <int BLOCK_THREADS>
__global__ void qubo_energy_kernel_optimized(
    const unsigned char* __restrict__ neighbours,  // N x n_items
    const double* __restrict__ Q,                  // n_items x n_items
    double* __restrict__ energies,                 // N
    int n_items) {
    int nbr = blockIdx.x;
    int tid = threadIdx.x;

    const unsigned char* x = neighbours + (size_t)nbr * n_items;

    double thread_sum = 0.0;

    for (int i = tid; i < n_items; i += BLOCK_THREADS) {
        if (x[i]) {
            const double* Qi = Q + (size_t)i * n_items;
            for (int j = 0; j < n_items; ++j) {
                if (x[j]) {
                    thread_sum += Qi[j];
                }
            }
        }
    }

    typedef hipcub::BlockReduce<double, BLOCK_THREADS> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    double block_sum = BlockReduce(temp_storage).Sum(thread_sum);

    if (tid == 0) {
        energies[nbr] = block_sum;
    }
}
