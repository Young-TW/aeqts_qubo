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
    // grid.y selects the neighbour; grid.x splits the (compacted) i-dimension
    // across multiple blocks so one neighbour's reduction is shared by several
    // CUs.
    //
    // The QUBO matrix Q is dense (4 MB, cache-resident and reused across all
    // neighbours), so this kernel is L2/compute-bound rather than VRAM-bound.
    // The energy only sums over pairs of *selected* items, so we first compact
    // the set-bit indices of x into shared memory (S[0..k-1]) and then run the
    // double loop over the k selected indices only. This turns the O(n^2) inner
    // loop -- which loaded x[j] and branched on it every iteration -- into a
    // branch-free O(k^2) loop of pure FMAs over Q. For this knapsack k ~ n/2.
    int nbr = blockIdx.y;
    int tid = threadIdx.x;

    const unsigned char* x = neighbours + (size_t)nbr * n_items;

    typedef hipcub::BlockScan<int, BLOCK_THREADS> BlockScan;
    typedef hipcub::BlockReduce<float, BLOCK_THREADS> BlockReduce;
    __shared__ union {
        typename BlockScan::TempStorage scan;
        typename BlockReduce::TempStorage reduce;
    } temp_storage;

    extern __shared__ int s_set[];  // n_items ints; holds the selected indices
    __shared__ int s_count;         // running number of selected items

    if (tid == 0) s_count = 0;
    __syncthreads();

    // Compact the set-bit indices of x into s_set in ascending order. A block
    // prefix-sum (rather than a shared atomic) is used so every block of this
    // neighbour produces the *identical* ordering -- the outer loop below is
    // tiled across grid.x blocks, so each block must agree on which compacted
    // position maps to which item.
    for (int base = 0; base < n_items; base += BLOCK_THREADS) {
        int j = base + tid;
        int flag = (j < n_items && x[j]) ? 1 : 0;
        int pos, total;
        BlockScan(temp_storage.scan).ExclusiveSum(flag, pos, total);
        if (flag) s_set[s_count + pos] = j;
        __syncthreads();
        if (tid == 0) s_count += total;
        __syncthreads();
    }
    int k = s_count;

    float thread_sum = 0.0f;

    int a_stride = gridDim.x * BLOCK_THREADS;
    for (int a = blockIdx.x * BLOCK_THREADS + tid; a < k; a += a_stride) {
        const float* Qi = Q + (size_t)s_set[a] * n_items;
        for (int b = 0; b < k; ++b) {
            thread_sum += Qi[s_set[b]];
        }
    }

    float block_sum = BlockReduce(temp_storage.reduce).Sum(thread_sum);

    // energies[] must be zeroed before launch; each i-tile adds its partial sum.
    if (tid == 0) {
        atomicAdd(&energies[nbr], block_sum);
    }
}
