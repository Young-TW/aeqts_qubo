# 效能分析與優化紀錄

## 環境

| 項目 | 值 |
|---|---|
| GPU | AMD Radeon RX 9070 XT (gfx1201, RDNA4, 64 CU) |
| Runtime | ROCm 7.13 / HIP backend |
| Profiler | rocprofv3 1.3.0 |
| 測試設定 | Items=500, N=50, Iter=1000, seed=12345 |
| 基準結果 | `Energy=-1.88399e+07 | Val=3094 | W=1374/1375 | VALID` |

## 如何重現 profiling

```bash
# 完整 system trace（kernel + HIP/HSA API + memory），輸出 CSV
rocprofv3 --sys-trace --stats --summary \
  -d prof/<config> -o report --output-format csv \
  -- ./build/aeqts_qubo 12345

# 只看 kernel 時間（調參時用，較快）
rocprofv3 --kernel-trace --stats \
  -d prof/<config> -o report --output-format csv \
  -- ./build/aeqts_qubo 12345
```

結果整理在 `prof/<config>/`（已加入 `.gitignore`，不入版控）。重點檔案：
`report_kernel_stats.csv`、`report_domain_stats.csv`、`report_hip_api_stats.csv`。

## 基準分析

### 時間分佈（report_domain_stats.csv）

| Domain | 時間 | 說明 |
|---|---|---|
| HIP_API | 228 ms | 含一次性冷啟動開銷 |
| HSA_API | 197 ms | runtime 底層等待 |
| **KERNEL_DISPATCH** | **169.5 ms** | 真正的 GPU 計算 |
| MEMORY_COPY | 0.05 ms | 僅 1 次 H2D（資料常駐 GPU）|

`hipDeviceSynchronize` 在 HIP API 佔 68%，但單次 max 154 ms 是 lazy context 初始化，**不隨迭代數增加**，可忽略。程式為 **kernel-bound**。

### Kernel 熱點（report_kernel_stats.csv，baseline）

| Kernel | 佔比 | 平均/次 |
|---|---|---|
| **`qubo_energy_kernel_optimized`** | **80.2%** | **135.7 µs** |
| rocprim radix_sort（能量排序）| 8.8% | 14.9 µs |
| `updateQ_kernel` | 5.3% | 8.9 µs |
| `generate_neighbours_kernel` | 4.0% | 6.7 µs |
| 其餘 | <2% | — |

能量計算 kernel 一個就吃掉 80% GPU 時間，是唯一值得優化的對象。

## 瓶頸：Grid 佔用率

原始 launch：`qubo_energy_kernel_optimized<256> <<<N, 256>>>`，即 **grid = 50 blocks**（1 block / neighbour）。

問題：

1. 50 blocks ≤ 64 CU，14 個 CU 全程閒置。
2. 更關鍵：每個 neighbour 的整份 O(n²)=250k 次 FP64 reduction 被綁在**單一 CU** 上序列跑完，kernel 的 critical path = 單一 block 的執行時間（≈135 µs）。

> FP64 為實驗需求，**不改精度**；改的是把同樣的運算量攤到更多 CU 平行做。

## 優化：沿 i 維度切塊（2D grid + partial reduce）

把「1 block = 1 neighbour」改成「`SPLIT` 個 block 協作 1 neighbour」：

- `grid = dim3(SPLIT, N)`：`blockIdx.y` 選 neighbour，`blockIdx.x` 切 i 維度。
- 每個 block 只處理一段 i（`i_start = blockIdx.x * blockDim.x + tid`，stride `gridDim.x * blockDim.x`），block 內 `BlockReduce` 後由 `tid==0` 做 `atomicAdd(&energies[nbr], block_sum)`。
- launch 前以 `hipMemsetAsync(d_energy, 0, ...)` 歸零（atomicAdd 前提）。

`atomicAdd` 呼叫數 = `N × SPLIT`（≤200），可忽略；歸零成本約 2.3 µs/代（1.7%）。

### 關鍵約束

tile 寬度 = block threads，所以 **`SPLIT × threads` 必須恰好覆蓋 `n_items`**，否則多出的 block 全空轉。
例：256-thread 下 `SPLIT>2` 只是新增空 block（`i_start ≥ 512 > 500`），`SPLIT=4` 反而因空 block 排程而變慢。要切更細必須縮小 block threads。

## 調參掃描（energy kernel 平均 GPU 時間，rocprofv3 實測）

| 配置 | working blocks | energy avg | 佔比 | 加速 |
|---|---|---|---|---|
| baseline `<256>` 1-way | 50 | 135729 ns | 80.2% | 1.00x |
| `<256>` 2-way | 100 | 98665 ns | 73.4% | 1.38x |
| `<256>` 3/5-way | — | ~99–100k | — | 僅加空 block，無增益 |
| **`<128>` 4-way** ✅ | **200** | **80839 ns** | **68.7%** | **1.68x** |
| `<64>` 8-way | 400 | 82023 ns | 68.6% | 1.65x（飽和回落）|

`<64>` 8-way 起 BlockReduce（block 太小）效率下降，故 **`<128>` + 4-way 為甜蜜點**。

## 結果

- **energy kernel：135.7 µs → 80.8 µs（1.68x）**，佔比 80.2% → 68.7%。
- 每代 wall-clock：0.185 ms → 0.132 ms（約 1.4x）。
- **數值完全一致**：seed 12345 / 9999 的 `Energy`、`Val`、`VALID` 與 baseline 相同（double 累加順序改變不影響能量排序）。

## 改動檔案

| 檔案 | 改動 |
|---|---|
| `src/hip/kernels.hpp`、`src/cuda/kernels.cuh` | kernel 改 2D grid + i-tile stride + `atomicAdd` |
| `src/hip/solver.hip`、`src/cuda/solver.cu` | `ENERGY_SPLIT=4`、`<128>` launch、launch 前歸零 |

- HIP 端已編譯並驗證。
- CUDA 端為對稱鏡像改動，**本機無 nvcc 未編譯**；不同 SM 數 / FP64 比例下最佳值會不同，需在 NVIDIA 環境重新調 `ENERGY_SPLIT`（維持 `ENERGY_SPLIT × threads ≥ n_items` 且 threads 不浪費）。

## 後續可優化方向

1. 仍以 energy kernel（68.7%）+ radix sort（~11%）為主。
2. 若 FP64 精度需求可放寬，將 `Q` 與累加改 FP32 — 在消費級 RDNA4 上 FP64 吞吐僅約 FP32 的 1/16，預期增益遠大於本次。
3. energy kernel 內層 `if(x[j])` 造成 warp divergence，可先壓縮 set-bit 索引把 O(n²) 降為 O(k²)。
