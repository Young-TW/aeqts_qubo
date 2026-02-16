#include "kernels.cuh"
#include <cmath>

__global__ void init_sequence_kernel(int* arr, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        arr[tid] = tid;
    }
}

__global__ void init_curand_states(curandStatePhilox4_32_10_t* states, unsigned long long seed, int total_threads)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_threads) return;
    curand_init(seed, tid, 0, &states[tid]);
}

__global__ void generate_neighbours_kernel(
    const float* __restrict__ q_alpha,
    const float* __restrict__ q_beta,
    unsigned char* __restrict__ neighbours, 
    int N,
    int n_items,
    curandStatePhilox4_32_10_t* states)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    int total = N * n_items;
    if (idx >= total) return;

    int nbr = idx / n_items;
    int i   = idx - nbr * n_items;

    curandStatePhilox4_32_10_t st = states[idx];
    float r = curand_uniform(&st); 
    states[idx] = st;

    float p1 = q_beta[i] * q_beta[i];
    neighbours[(size_t)nbr * n_items + i] = (r > p1) ? 0 : 1;
}

__global__ void updateQ_kernel(
    const unsigned char* __restrict__ neighbours, 
    const int* __restrict__ sorted_idx,           
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

        int diff = (int)xb - (int)xw; 
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

__global__ void update_global_best_kernel(
    const float* __restrict__ sorted_energies,
    const int* __restrict__ sorted_idx,
    const unsigned char* __restrict__ neighbours,
    float* __restrict__ global_best_energy,
    unsigned char* __restrict__ global_best_sol,
    int n_items)
{
    if (blockIdx.x == 0) {
        float current_best = sorted_energies[0];
        if (current_best < *global_best_energy) {
            if (threadIdx.x == 0) {
                *global_best_energy = current_best;
            }
            int best_idx = sorted_idx[0];
            for (int i = threadIdx.x; i < n_items; i += blockDim.x) {
                global_best_sol[i] = neighbours[(size_t)best_idx * n_items + i];
            }
        }
    }
}
