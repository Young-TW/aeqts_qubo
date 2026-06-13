#ifndef AEQTS_SOLVER_H
#define AEQTS_SOLVER_H

#include <vector>

// 主機端與 GPU 後端之間的純 C++ 介面。
// 實作位於各後端的 device 原始碼(src/cuda/solver.cu 或 src/hip/solver.hip),
// 入口 main.cpp 不需引用任何 CUDA/HIP 標頭。

struct AeqtsParams {
    int iter;                    // 迭代代數
    int n_items;                 // 物品數量 (問題維度)
    int N;                       // 種群大小 (量子染色體數)
    unsigned long long seed;     // 實際使用的亂數種子
};

struct AeqtsResult {
    float best_energy;                         // 找到的最低能量 (FP32)
    std::vector<unsigned char> best_solution;  // 長度 n_items 的 0/1 解
    double avg_iter_ms;                        // 每代平均耗時 (毫秒,計時)
};

// 依本 MPI local rank 綁定 GPU 裝置(單卡情況下等同選用裝置 0)。
void gpu_set_device(int local_rank);

// 在 GPU 上執行完整的 AEQTS + QUBO 搜尋。
// Qh 為 row-major 的 n_items x n_items 主機端 QUBO 矩陣 (FP32)。
AeqtsResult run_aeqts(const AeqtsParams& params, const std::vector<float>& Qh);

#endif  // AEQTS_SOLVER_H
