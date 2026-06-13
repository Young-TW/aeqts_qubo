# ROCm 效能分析（Profiling）指南

本專案在 AMD GPU（HIP backend）上的效能分析工具與使用方式。
分析「結果與優化紀錄」請見 [`docs/performance.md`](../performance.md)；本文只講**怎麼量**。

## 環境

| 項目 | 值 |
|---|---|
| GPU | AMD Radeon RX 9070 XT (gfx1201, RDNA4, 64 CU) |
| Runtime | ROCm 7.13 / HIP backend |
| Profiler | rocprofv3 1.3.0 |
| 慣用設定 | seed=12345（與既有報告一致）|

所有 profiling 輸出落在 `prof/<config>/`，已加入 `.gitignore`、不入版控；
渲染器與腳本本身入版控可重用。

---

## 重要前提：這張卡量得到什麼、量不到什麼

> **計時（kernel 花多久）和硬體計數器（kernel 內部在幹嘛）是兩套獨立硬體。**

| 類型 | 機制 | gfx1201 狀態 |
|---|---|---|
| **計時 / trace** | GPU dispatch 的 start/end 時間戳 | ✅ 完全可用 |
| **硬體計數器 PMC** | CU 內部事件計數器（VALU 指令、記憶體 byte、cache 命中…） | ⚠️ 大多回 0 |

實測在 gfx1201 / ROCm 7.13 下，**只有週期級 counter 有值**：

- ✅ 有值：`GRBM_COUNT`、`GRBM_GUI_ACTIVE`、`SQ_BUSY_CYCLES`、`SQ_WAVES`
- ❌ 全回 0：`SQ_INSTS_VALU/SALU`（指令數）、`TA/TCP/TCC/GL2C`（記憶體/快取/頻寬）、
  `MeanOccupancyPerCU`、`L2CacheHit`、`VALUBusy`、`MemUnitBusy` …

**這也是為什麼 omniperf（已改名 `rocprof-compute`）在這張卡上幫不上忙**：

1. `rocprof-compute` 沒有 gfx1201 的 SoC 設定檔（只支援 CDNA MI 系列 + gfx1151），裝了也跑不起來；
2. 更根本的是，它的 roofline / speed-of-light / memory-chart 全靠上面那些「回 0」的 PMC 當原料。
   底層硬體沒把這些計數器接出來，換哪個前端工具都一樣。

**結論**：要判斷「這個 kernel 慢」用計時即可；要判斷「為什麼慢——卡在算術還是記憶體頻寬」，
在這張卡上量不到，只能靠 kernel-trace 計時 + 原始碼推理。並行度/佔用率天花板則可由
`SQ_WAVES` + grid size 推得（見下）。

---

## 工具總覽

| 工具 | 回答的問題 | 在 gfx1201 |
|---|---|---|
| `rocprofv3 --kernel-trace` | 各 kernel 花多久、佔比多少 | ✅ 主力 |
| `scripts/fold_to_flamegraph.py` | 時間分佈的火焰圖視覺化 | ✅ |
| `scripts/profile_counters.sh` | 並行度/佔用率（週期級 counter） | ⚠️ 僅週期級 |

---

## 1. kernel 計時與 trace（rocprofv3）

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

重點輸出檔：`report_kernel_stats.csv`（kernel 熱點）、`report_domain_stats.csv`（時間分佈）、
`report_hip_api_stats.csv`（HIP API）。

也可改 `--output-format pftrace` 重跑，把結果丟進 [Perfetto](https://ui.perfetto.dev) 看時間軸。

## 2. 火焰圖（fold_to_flamegraph.py）

GPU kernel 沒有 host 呼叫堆疊，故以 **kernel 總 GPU 時間為寬度**，建成三層
`aeqts_qubo → 階段(Energy/Measure/Sort/Update) → kernel` 的火焰圖。
`scripts/fold_to_flamegraph.py` 為純 Python、無外部依賴的渲染器：

```bash
# folded stacks 由 report_kernel_trace.csv 逐列 End-Start 加總而得（需先跑過 --kernel-trace）
python3 scripts/fold_to_flamegraph.py \
  prof/<config>/gpu.folded prof/<config>/flamegraph.svg \
  "標題" "副標題"
```

產出的 SVG 每個 box 有 hover tooltip 顯示 ns / %。

## 3. 硬體計數器（profile_counters.sh + summarize_counters.py）

對指定 kernel 採集週期級 counter，彙整成可讀表格並推導並行度。
單一 pass、無 replay；只收實際有值的 counter，不以零值誤導判讀。

```bash
scripts/profile_counters.sh                 # 預設只測 energy kernel
scripts/profile_counters.sh ".*"            # 測全部 kernel
KERNEL=updateQ ITER=200 SEED=42 scripts/profile_counters.sh
```

環境變數：`KERNEL`（kernel 名稱 regex）、`ITER`、`SEED`、`BIN`、`OUTDIR`。

輸出範例（energy kernel）：

```
━━ qubo_energy_kernel_optimized<128>  (201 次 dispatch 平均) ━━
  GPU 活躍週期 (GRBM_GUI_ACTIVE)          352,010  ∝ kernel 執行時間
  GPU 活躍佔比                            100.0%  活躍週期 / 總週期
  SQ 忙碌週期 (SQ_BUSY_CYCLES)         11,227,352  指令仲裁器工作週期(跨 SIMD 累加)
  SQ_WAVES / dispatch                      3,964  SQ 計到的 wave 數
  平均 grid size                          32,768  threads
  grid 推算 wave 數                        1,024  grid / wavefront(32)
  → grid 32,768 threads ÷ wave32 ÷ 64 CU ≈ 16.0 wave/CU 同時可駐留。
```

每 CU 16 wave（理論上限 32）即約 50% 佔用天花板，印證了下節「grid 佔用率不足」的判斷。

---

## 基準熱點摘要

以 `rocprofv3 --kernel-trace` 量測（seed=12345），GPU 時間分佈：

| Kernel | 佔比 | 平均/次 |
|---|---|---|
| **`qubo_energy_kernel_optimized`** | **~80%（baseline）** | 135.7 µs |
| rocprim radix_sort（能量排序） | ~9% | 14.9 µs |
| `updateQ_kernel` | ~5% | 8.9 µs |
| `generate_neighbours_kernel` | ~4% | 6.7 µs |

能量計算 kernel 一個就吃掉約 80% GPU 時間，是唯一值得優化的對象。
程式為 **kernel-bound**（資料一次上 GPU 後常駐，幾乎無搬移）。

完整的瓶頸分析與優化紀錄（grid 切塊、FP32、調參掃描、結果）見
[`docs/performance.md`](../performance.md)。
