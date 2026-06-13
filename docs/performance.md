# 效能分析與優化紀錄

本文聚焦**量測結果與優化決策**。profiling 工具(rocprofv3 計時/trace、火焰圖、
硬體計數器)的操作方式見 [`docs/rocm/profiling.md`](rocm/profiling.md)。

## 現況摘要(TL;DR)

- 程式為 **kernel-bound**,`qubo_energy_kernel_optimized`(計算 E = xᵀQx)是唯一熱點。
- 已落地兩項優化:**2D grid + i 維度切塊**(1.68x)與**全程 FP32 能量路徑**(energy
  kernel 1.30x)。目前 energy kernel 約佔 GPU 時間 65%。
- 此 kernel 已位於**結構甜蜜點**:其瓶頸會隨 `n_items` 翻轉(小 n 受 cache/compute 限、
  大 n 受 VRAM 頻寬限),因此再做單一改寫很難「通用」。三個自然的改寫(欄平行
  coalescing、set-bit 壓縮、對稱上三角)經 A/B 實測**都只在某個區間有效、在另一區間回歸**,
  已全部否決,維持 baseline。見〈已否決的「不通用」最佳化〉。
- 任何後續最佳化都須以 [`scripts/validate_perf.py`](../scripts/validate_perf.py)
  跨尺寸驗證「不回歸」才採納。

## 環境

| 項目 | 值 |
|---|---|
| GPU | AMD Radeon RX 9070 XT (gfx1201, RDNA4, 64 CU) |
| Runtime | ROCm 7.13 / HIP backend |
| Profiler | rocprofv3 1.3.0 |
| 早期 micro-benchmark 設定 | Items=500, N=50, Iter=1000, seed=12345(下方 rocprofv3 數據沿用)|
| 跨尺寸驗證設定 | N=64,n_items ∈ {1000…4000}(由 `validate_perf.py` 掃描)|

> 下方〈優化歷程〉的 rocprofv3 絕對數字是在早期 500/50/1000 設定下量的;
> 跨尺寸的相對效能改用驗證腳本量(見最後一節)。兩者目的不同:前者看單 kernel
> 微觀時間,後者看不同問題規模下是否整體不回歸。

## 基準分析(FP64 時代,Items=500/N=50/Iter=1000)

時間分佈顯示程式 **kernel-bound**:`hipDeviceSynchronize` 雖在 HIP API 佔比高,但其
單次 max ~154 ms 是 lazy context 初始化,**不隨迭代數增加**,可忽略。真正的 GPU 計算
(KERNEL_DISPATCH)約 169.5 ms,而 H2D 只有 0.05 ms(資料常駐 GPU)。

Kernel 熱點(`report_kernel_stats.csv`,FP64 baseline):

| Kernel | 佔比 | 平均/次 |
|---|---|---|
| **`qubo_energy_kernel_optimized`** | **80.2%** | **135.7 µs** |
| rocprim radix_sort(能量排序)| 8.8% | 14.9 µs |
| `updateQ_kernel` | 5.3% | 8.9 µs |
| `generate_neighbours_kernel` | 4.0% | 6.7 µs |
| 其餘 | <2% | — |

energy kernel 一個就吃 80% GPU 時間,是唯一值得優化的對象。

## 優化歷程

### 1. Grid 佔用率:2D grid + i 維度切塊(1.68x)

原始 launch 為 `<<<N, 256>>>`(1 block / neighbour),即 50 blocks。問題:

1. 50 blocks ≤ 64 CU,14 個 CU 全程閒置。
2. 更關鍵:每個 neighbour 的整份 O(n²) reduction 被綁在**單一 CU** 序列跑完,
   critical path = 單一 block 的執行時間。

改為「`SPLIT` 個 block 協作 1 neighbour」:

- `grid = dim3(SPLIT, N)`:`blockIdx.y` 選 neighbour、`blockIdx.x` 切 i 維度
  (`i_start = blockIdx.x*blockDim.x + tid`,stride `gridDim.x*blockDim.x`)。
- block 內 `BlockReduce` 後由 `tid==0` 做 `atomicAdd(&energies[nbr], block_sum)`;
  launch 前以 `hipMemsetAsync(d_energy, 0, …)` 歸零。`atomicAdd` 數 = `N×SPLIT`(≤200)、
  歸零約 2.3 µs/代(1.7%),皆可忽略。

**關鍵約束**:tile 寬度 = block threads,所以 `SPLIT × threads` 須恰好覆蓋 `n_items`,
否則多出的 block 全空轉(256-thread 下 `SPLIT=4` 反因空 block 排程變慢)。要切更細須縮小
block threads。調參掃描(energy kernel 平均 GPU 時間,rocprofv3 實測):

| 配置 | working blocks | energy avg | 加速 |
|---|---|---|---|
| baseline `<256>` 1-way | 50 | 135729 ns | 1.00x |
| `<256>` 2-way | 100 | 98665 ns | 1.38x |
| `<256>` 3/5-way | — | ~99–100k | 僅加空 block,無增益 |
| **`<128>` 4-way** ✅ | **200** | **80839 ns** | **1.68x** |
| `<64>` 8-way | 400 | 82023 ns | 1.65x(飽和回落)|

`<64>` 8-way 起 BlockReduce(block 太小)效率下降,故 **`<128>` + 4-way 為甜蜜點**
(現行 `ENERGY_SPLIT=4`)。結果:energy kernel 135.7→80.8 µs(1.68x);數值與 baseline
完全一致(累加順序改變不影響能量排序)。

### 2. 全程 FP32 能量路徑(energy kernel 1.30x)

把能量相關的整條資料路徑由 `double` 改 `float`,並讓 **host→solver 介面全程 float**
(float 進 float 出,無 double↔float 轉換邊界):`qubo_energy_kernel` /
`update_global_best_kernel` / solver 緩衝區 / `hipcub::DeviceRadixSort` 鍵 /
`build_teacher_qubo_matrix_host` / `main.cpp`(含 MPI `MPI_FLOAT_INT`)全改 float,
僅計時變數保留 double。

rocprofv3 同機重新量測(`<128>` 4-way、seed=12345、Iter=1000):

| Kernel | FP64 avg | FP32 avg | 加速 |
|---|---|---|---|
| **`qubo_energy_kernel`** | **80.5 µs** (68.9%) | **61.9 µs** (65.4%) | **1.30x** |
| rocprim radix_sort | 14.9 µs (12.8%) | 8.7 µs (9.2%) | **1.71x** |
| updateQ_kernel | 8.8 µs | 9.0 µs | ~1.0x |
| generate_neighbours | 7.0 µs | 7.1 µs | ~1.0x |
| **kernel 總時間** | **117.0 ms** | **94.7 ms** | **1.23x** |
| 每代 wall-clock(AvgIter)| 0.132 ms | 0.111 ms | 1.19x |

- **energy kernel 1.30x**:增益主要來自 `Q` 記憶體流量減半(8→4 bytes/元素),而非
  FP64 ALU 懲罰——此 kernel 內層只有加法、高度 memory-bound,故非「數倍」但 1.30x 穩定。
- **radix sort 1.71x**:排序鍵位元數減半,工作量直接下降。

**精度影響**:能量(6 位有效數字)幾乎一致、結果仍 VALID,但 `Val`/`W` 略不同。原因:
能量量級約 1.88e7,float 在此量級解析度約 ±2,使多個近乎等能量的解排序改變、挑到的最佳解
略異。**若需精確重現/排序穩定性應保留 FP64;以吞吐為優先且可接受 ~1e-6 相對誤差則 FP32 划算。**

| | seed 12345 | seed 9999 |
|---|---|---|
| FP64 | `Energy=-1.88399e+07, Val=3094, W=1374/1375, VALID` | `VALID` |
| FP32 | `Energy=-1.88399e+07, Val=3086, W=1371/1375, VALID` | `Energy=-1.88397e+07, Val=3081, W=1371/1375, VALID` |

## 已否決的「不通用」最佳化

> 原則:**只接受跨參數區間都不回歸的最佳化**;只在特定 n 區間才快、換區間就變慢的不採納。

energy kernel(現 ~65%)的瓶頸**隨 `n_items` 翻轉**:

- 小 n(如 1000):`Q`(n²×4 = 4 MB)常駐 cache → cache/compute-bound。
- 大 n(如 4000,實際 config):`Q` = 64 MB,超出 cache → VRAM 頻寬 bound。

baseline 之所以快,在於**每列連續 streaming 讀取**(每 thread 擁有一列、`Qi[j]` 隨 j++
連續讀),prefetcher 很吃這套。下列三個改寫各自破壞了這個特性或只在單一區間有效,經
`validate_perf.py` A/B 實測(HIP / gfx1201,加速比 = baseline÷候選,>1 才是變快):

| 候選改寫 | n=1000 | n=2000 | n=3000 | n=4000 | 結論 |
|---|---|---|---|---|---|
| 欄平行 coalescing(lane 讀同列相鄰 j)| 0.60x | 0.69x | 0.58x | 0.69x | 全面回歸 |
| 一次性 set-bit 壓縮(O(k²),inner `Qi[sidx[cj]]`)| 0.98x | 0.68x | 0.60x | 0.61x | 大 n 回歸 |
| 對稱上三角(掃 j>i、對角一次+非對角×2)| 0.69x | 0.84x | 0.93x | **1.16x** | 區間特定 |

- **欄平行**:逼每個 thread 跑完整圈外層 i 迴圈(n 次 `x[i]` 檢查)卻只做極少內層工作,
  冗餘外層掃描主導 → 全面變慢。
- **壓縮**:`Qi[sidx[cj]]` 的位址相依於剛載入的索引 → gather 打斷 streaming/prefetch;
  小 n 被 cache 掩蓋(~中性),大 n 致命。
- **對稱**:流量減半(大 n 贏),但三角迴圈的資料相依 trip count 造成 warp divergence
  (小 n 輸)——典型的區間特定取捨。

**結論**:baseline 落在結構甜蜜點,無簡單通用改寫。唯一**可能**通用的方向是
**讓載入的每列 `Q` 跨多個 neighbour 重複使用**(在大 n 攤掉 VRAM 流量、又不動小 n 的存取
樣式),屬大型改寫、尚未驗證。在大小 n 都實測不回歸前,維持 baseline。

## 驗證方法:`scripts/validate_perf.py`

判斷一項最佳化是否「通用」的 A/B harness:

- 掃多組 `(n_items, iter, N)` case,涵蓋兩種瓶頸區間。
- **交錯**輪流跑各 binary 並先暖機,讓兩版吃到相同 GPU 時脈狀態(避開閒置降頻:mclk 會
  掉到 ~772 MHz);對 `--reps` 輪取**中位數**而非平均。
- 任一 case 慢於基準超過 `--rtol`(預設 2%)即標 REGRESSION;另比較能量相對誤差
  (`--etol`)。有回歸或能量不符時 exit code ≠ 0,可當 CI 關卡。
- **量測注意**:大 n(≥4000)單測抖動約 ±5%,須用 `--reps>=3`(更大 n 再加)才壓得住雜訊。

```bash
# 基準掃描
scripts/validate_perf.py --bin build/aeqts_qubo:baseline --save baseline.json
# A/B:baseline vs 候選版
scripts/validate_perf.py --bin build_base/aeqts_qubo:base --bin build_opt/aeqts_qubo:opt --reps 3
```

## 後續可優化方向

1. **energy kernel(~65%)**:已知三個簡單改寫不通用(上表)。下一步只剩「跨 neighbour
   重用 `Q` 列」這類較大改寫,且須先過 `validate_perf.py` 才採納。
2. **radix sort(~9%)**:鍵已是 float、N 小,空間有限。
3. **高精度能量回報**:可維持 GPU FP32 計算,僅在最終 `best_energy` 以 Kahan/double 重算
   一次選定解(成本一次,不影響迴圈吞吐)。

## 移植注意

- HIP 端已編譯、執行並以 rocprofv3 驗證(數據如上);CUDA 端為對稱鏡像改動,**本機無 nvcc 未編譯**。
- 不同 SM 數 / FP64 比例下 `ENERGY_SPLIT` 最佳值會不同,需在 NVIDIA 環境重新調(維持
  `ENERGY_SPLIT × threads ≥ n_items` 且 threads 不浪費)。
