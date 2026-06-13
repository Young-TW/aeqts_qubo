# 效能分析與優化紀錄

## 環境

| 項目 | 值 |
|---|---|
| GPU | AMD Radeon RX 9070 XT (gfx1201, RDNA4, 64 CU) |
| Runtime | ROCm 7.13 / HIP backend |
| Profiler | rocprofv3 1.3.0 |
| 測試設定 | Items=500, N=50, Iter=1000, seed=12345 |
| 基準結果 | `Energy=-1.88399e+07 | Val=3094 | W=1374/1375 | VALID` |

> profiling 工具（rocprofv3 計時/trace、火焰圖、硬體計數器）的使用方式見
> [`docs/rocm/profiling.md`](rocm/profiling.md)。本文聚焦於量測結果與優化紀錄。

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
2. ~~若 FP64 精度需求可放寬，將 `Q` 與累加改 FP32~~ → 已實作,見下節。
3. energy kernel 內層 `if(x[j])` 造成 warp divergence，可先壓縮 set-bit 索引把 O(n²) 降為 O(k²)。

---

# FP32 能量計算（接續優化）

> 前一節保留 FP64 為「實驗需求」。本節依需求把能量計算路徑全面改為 FP32 並實測效能,
> 代價是回報能量的精度下降(見下「精度影響」)。

## 改動

能量相關的整條資料路徑由 `double` 改為 `float`,並進一步讓 **host 端到 solver 介面全程 float**
(float 進 float 出,不再有任何 double↔float 轉換邊界):

- `qubo_energy_kernel_optimized`:`Q`、`thread_sum`、`BlockReduce`、`energies`、`atomicAdd` 全部 FP32。
- `update_global_best_kernel`:`sorted_energies`、`global_best_energy` 改 FP32。
- `solver`:`dQ`、`d_energy`、`d_energy_sorted`、`d_global_best_energy` 改 FP32;
  `hipcub::DeviceRadixSort` 因鍵型別變 float 自動改排 float。
- **介面 `solver.h`**:`run_aeqts` 收 `std::vector<float>` Qh、`AeqtsResult::best_energy` 為 `float`
  (`avg_iter_ms` 為計時,保留 double)。`Qh` 為 float 直接上傳、`best_energy` 為 float 直接回傳,
  省掉先前的 host 端轉換。
- **`build_teacher_qubo_matrix_host`**:輸入/輸出全改 `float`。
- **`main.cpp`**:weights / values / capacity / penalty / 能量彙整(含 MPI `MPI_FLOAT_INT` MINLOC)
  全部 `float`,僅計時變數 `avg_ms` 保留 `double`。

## rocprofv3 對比（同機重新量測,`<128>` 4-way 配置）

兩版皆以 `rocprofv3 --kernel-trace --stats` 量測 seed=12345、Iter=1000,於同一機台連續跑：

```bash
rocprofv3 --kernel-trace --stats -d prof/fp64_split4_128thr -o report --output-format csv -- ./build/aeqts_qubo 12345
rocprofv3 --kernel-trace --stats -d prof/fp32_split4_128thr -o report --output-format csv -- ./build/aeqts_qubo 12345
```

| Kernel | FP64 avg | FP32 avg | 加速 |
|---|---|---|---|
| **`qubo_energy_kernel`** | **80.5 µs** (68.9%) | **61.9 µs** (65.4%) | **1.30x** |
| rocprim radix_sort | 14.9 µs (12.8%) | 8.7 µs (9.2%) | **1.71x** |
| updateQ_kernel | 8.8 µs | 9.0 µs | ~1.0x |
| generate_neighbours | 7.0 µs | 7.1 µs | ~1.0x |
| **kernel 總時間** | **117.0 ms** | **94.7 ms** | **1.23x** |
| 每代 wall-clock（AvgIter）| 0.132 ms | 0.111 ms | 1.19x |

兩個吃時間的 kernel 都受益:

- **energy kernel 1.30x**：增益主要來自記憶體流量減半(`Q` 由 8 bytes/元素降到 4),而非 FP64 ALU 懲罰。此 kernel 內層只有加法(無乘法)且高度 memory-bound,因此沒有達到「數倍」的樂觀預期,但 1.30x 為實測穩定值。
- **radix sort 1.71x**:排序鍵由 double 變 float,位元數減半,排序工作量直接下降。

## 精度影響

| | seed 12345 | seed 9999 |
|---|---|---|
| FP64 | `Energy=-1.88399e+07 | Val=3094 | W=1374/1375 | VALID` | `VALID` |
| FP32 | `Energy=-1.88399e+07 | Val=3086 | W=1371/1375 | VALID` | `Energy=-1.88397e+07 | Val=3081 | W=1371/1375 | VALID` |

顯示能量(到 6 位有效數字)幾乎一致、結果仍 **VALID**,但 `Val`/`W` 略有不同。原因:能量量級約 1.88e7,
float 在此量級的解析度約 ±2,使得多個能量近乎相等的解之間的排序結果改變,挑到的最佳解略有差異。
**若需精確重現或排序穩定性,應保留 FP64;若以吞吐為優先且可接受能量 ~1e-6 相對誤差,FP32 划算。**

## 改動檔案

| 檔案 | 改動 |
|---|---|
| `src/hip/kernels.hpp`、`src/cuda/kernels.cuh` | energy kernel 與 `update_global_best_kernel` 介面改 FP32 |
| `src/hip/kernels.hip`、`src/cuda/kernels.cu` | `update_global_best_kernel` 內部改 FP32 |
| `src/hip/solver.hip`、`src/cuda/solver.cu` | 能量路徑緩衝區改 FP32;`Qh` float 直接上傳、`best_energy` float 直接回傳(無轉換) |
| `include/solver.h` | `run_aeqts` 收 `vector<float>`、`AeqtsResult::best_energy` 為 `float`(`avg_iter_ms` 計時仍 double)|
| `include/qubo_matrix.h`、`src/qubo_matrix.cpp` | `build_teacher_qubo_matrix_host` 輸入/輸出全改 `float` |
| `src/main.cpp` | weights/values/capacity/penalty/能量彙整(`MPI_FLOAT_INT`)改 `float`,僅計時保留 double |
| `test/test_qubo_matrix.cpp` | 改 `float` 向量、容差放寬到 float 等級 |

- HIP 端已編譯、執行並以 rocprofv3 驗證(數據如上)。
- CUDA 端為對稱鏡像改動,**本機無 nvcc 未編譯**。

## 後續可優化方向(更新)

1. ~~energy kernel 可壓縮 set-bit 索引把內層 `if(x[j])` 的 O(n²) 降為 O(k²)~~ → 已實作,見下節。
2. radix sort(~9%)鍵已是 float、N=50 很小,空間有限。
3. 若需更高精度的能量回報,可在 GPU 端維持 FP32 計算、僅在最終 `best_energy`
   以 Kahan/double 重算一次選定解的能量(成本一次,不影響迴圈吞吐)。

---

# Set-bit 壓縮:O(n²) → O(k²)(接續優化)

> 量測設定改為 `config/case.conf` 現值:**Items=1000, N=64, Iter=1000, seed=12345**
> (與前幾節 Items=500 不同,故下方 baseline 為「同設定重新量測」的值,不可與前節 µs 直接比較)。

## 動機:先排除一個直覺上的錯誤方向

直覺會認為這個 memory-bound kernel 慢在 **`Q` 讀取未 coalesce**——原本一個 thread 負責一段
row `i`,warp 內相鄰 lane 讀的是相鄰 *row*(位址 stride = `n_items×4` bytes),完全不 coalesce。
照此把平行維度改成沿 column `j`(讓 lane 讀相鄰 column → coalesced)後,**實測反而變慢**
(每代 0.24 → 0.35 ms)。

原因:`Q` 只有 `n²×4 = 4 MB`,完整放得進 L2 / Infinity Cache,且**每代被 64 個 neighbour 重複讀取**。
因此 kernel 實為 **cache/compute-bound 而非 VRAM 頻寬 bound**,coalesce 帶來的頻寬節省被 cache 命中
掩蓋;column 平行版反而因「每 thread 內層只剩 ~8 圈、外層 row 序列化」而吃了迴圈控制與延遲的虧。

## 改法:壓縮被選中索引

能量只在「兩個都被選中」的 item 對上累加。先用 **block 前綴和(`BlockScan::ExclusiveSum`)**
把 `x` 的 set-bit 索引壓進 shared memory `s_set[0..k-1]`(升序),再對這 `k` 個索引跑雙重迴圈:

```cpp
for (int a = blockIdx.x*BLOCK + tid; a < k; a += gridDim.x*BLOCK) {
    const float* Qi = Q + (size_t)s_set[a] * n_items;
    for (int b = 0; b < k; ++b) thread_sum += Qi[s_set[b]];  // 無分支、無 x[j] 載入
}
```

把內層由「掃滿 `n_items`、每圈 `if(x[j])`」變成 **branch-free 的 O(k²) 純 FMA**。此 knapsack
capacity = 半總重,`k ≈ n/2`,故對數約減半,並省掉每圈的 `x[j]` 載入與 warp 分歧。

### 關鍵正確性陷阱(踩過)

`grid.x`(`ENERGY_SPLIT`)會把**外層 `a` 切給多個 block 協作同一 neighbour**,所以每個 block 的
`s_set` 排序**必須完全一致**,各 block 的 `a`-分段才能恰好拼成完整集合。
一開始用 shared-memory `atomicAdd` 來 append,**各 block 得到不同的排列** → `a` 分段重疊/漏算,
結果 `OVERWEIGHT`、能量系統性偏低。改用 `BlockScan` 升序壓縮(確定性、跨 block 一致)後正確。
`ENERGY_SPLIT=1` 時剛好無此問題,故 debug 時以 `SPLIT=1` 對照可快速定位。

## rocprofv3 對比(同機,Items=1000/N=64,`--kernel-trace --stats`,seed=12345)

| Kernel | baseline avg | 壓縮後 avg | 加速 |
|---|---|---|---|
| **`qubo_energy_kernel`** | **188.2 µs**(84.2%) | **146.6 µs**(79.9%) | **1.28x** |
| 每代 wall-clock(AvgIter) | 0.239 ms | 0.20 ms | ~1.2x |

加速「只有」1.28x 而非 ~2x:剩下的瓶頸仍是 `Qi[s_set[b]]` 的讀取——warp 內 lane 的 row 不同、
column 相同,依然 strided(非 coalesce),只是次數由 n 降到 k。`ENERGY_SPLIT` 在 2/4/6/8 掃描下
**4-way 仍為甜蜜點**。

## 數值與驗證

- seed 12345/9999 皆 **VALID**,能量到 6 位有效數字與 baseline 一致(`-7.5489e+07`);
  `Val`/`W` 的微小差異與前節 FP32 累加順序敏感性同源。
- `ctest`(qubo_matrix / config)通過。

## 改動檔案

| 檔案 | 改動 |
|---|---|
| `src/hip/kernels.hpp`、`src/cuda/kernels.cuh` | energy kernel 改為 `BlockScan` 壓縮 set-bit + O(k²) 雙迴圈 |
| `src/hip/solver.hip`、`src/cuda/solver.cu` | launch 加上 dynamic shared mem `n_items*sizeof(int)`;移除舊的 SPLIT 覆蓋約束註解 |

- HIP 端已編譯、執行並以 rocprofv3 驗證(數據如上)。
- CUDA 端為對稱鏡像改動,**本機無 nvcc 未編譯**。

## 後續可優化方向(再更新)

1. energy kernel 仍是最大宗且 `Q` 讀取仍未 coalesce。要再進一步需做 **2D tiling**(一個 warp 負責
   一塊 row×col 子矩陣的 reduction,類 GEMM),讓 `Q` 讀取同時兼顧 coalesce 與 cache 重用——
   屬較大改寫。
2. 利用 `Q` 對稱性只算上三角(`E = diag·x + 2·Σ_{i<j}`)可再砍一半 pair 運算,但改變 FP32 累加
   順序、影響排序穩定性,建議作為可選項。
3. radix sort 與精度重算同前節。
