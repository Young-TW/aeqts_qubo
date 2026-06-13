#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <cub/cub.cuh>

constexpr float PI_F = 3.14159265358979323846f;

// ------------------------------ 巨集 ------------------------------
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                               \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                         cudaGetErrorString(_e));                              \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

// ------------------------------ Kernel 宣告 ------------------------------

__global__ void init_sequence_kernel(int* arr, int n);

__global__ void init_curand_states(curandStatePhilox4_32_10_t* states,
                                   unsigned long long seed, int total_threads);

__global__ void generate_neighbours_kernel(
    const float* __restrict__ q_alpha, const float* __restrict__ q_beta,
    unsigned char* __restrict__ neighbours,  // N x n_items (0/1)
    int N, int n_items, curandStatePhilox4_32_10_t* states);

__global__ void updateQ_kernel(
    const unsigned char* __restrict__ neighbours,  // N x n_items
    const int* __restrict__ sorted_idx,            // N (ascending energies)
    float* __restrict__ q_alpha, float* __restrict__ q_beta, int N,
    int n_items);

__global__ void update_global_best_kernel(
    const float* __restrict__ sorted_energies,
    const int* __restrict__ sorted_idx,
    const unsigned char* __restrict__ neighbours,
    float* __restrict__ global_best_energy,
    unsigned char* __restrict__ global_best_sol, int n_items);

// ------------------------------ Template Kernel 實作
// ------------------------------ 注意：樣板函式必須實作在標頭檔中
template <int BLOCK_THREADS>
__global__ void qubo_energy_kernel_optimized(
    const unsigned char* __restrict__ neighbours,  // N x n_items
    const float* __restrict__ Q,                   // n_items x n_items
    float* __restrict__ energies,                  // N
    int n_items) {
    // grid.y selects the neighbour; grid.x splits the i-dimension across
    // multiple blocks so one neighbour's reduction is shared by several SMs.
    int nbr = blockIdx.y;
    int tid = threadIdx.x;

    const unsigned char* x = neighbours + (size_t)nbr * n_items;

    float thread_sum = 0.0f;

    int i_start = blockIdx.x * BLOCK_THREADS + tid;
    int i_stride = gridDim.x * BLOCK_THREADS;
    for (int i = i_start; i < n_items; i += i_stride) {
        if (x[i]) {
            const float* Qi = Q + (size_t)i * n_items;
            for (int j = 0; j < n_items; ++j) {
                if (x[j]) {
                    thread_sum += Qi[j];
                }
            }
        }
    }

    typedef cub::BlockReduce<float, BLOCK_THREADS> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    float block_sum = BlockReduce(temp_storage).Sum(thread_sum);

    // energies[] must be zeroed before launch; each i-tile adds its partial sum.
    if (tid == 0) {
        atomicAdd(&energies[nbr], block_sum);
    }
}
