rocprofv3 效能報告：`./build/aeqts_qubo`

**環境**：AMD Radeon RX 9070 XT (gfx1201, RDNA4) · ROCm 7.13 · rocprofv3 1.3.0
**設定**：Items=500, N=50, Iter=1000, seed=12345
**產出**：`prof/report_*.csv`（kernel/HIP/HSA trace + stats）

---

### 1. 整體時間分佈（domain_stats）

| Domain | 總時間 | 說明 |
|---|---|---|
| HIP_API | 228 ms | **含一次性啟動開銷**（見下） |
| HSA_API | 197 ms | runtime 底層，多為等待 |
| **KERNEL_DISPATCH** | **169.5 ms** | 真正的 GPU 計算時間 |
| MEMORY_ALLOCATION | 2.3 ms | 16 次 malloc |
| MEMORY_COPY | 0.05 ms | 僅 1 次 H2D (49 µs) |

**關鍵觀察**：幾乎沒有資料搬移（只有 1 次 H2D 49µs），這支程式是 **kernel-bound**，資料一次上 GPU 後就常駐。瓶頸完全在 compute kernel。

### 2. 啟動開銷 vs 穩態（HIP API stats）

`hipDeviceSynchronize` 佔 HIP API 68% (155 ms)，但其中**單次 max = 154.8 ms** 是 lazy context 初始化的一次性成本；`hipGetDeviceCount` 8.9ms、第一次 `hipMemcpy` 21ms 同樣是冷啟動。這些**不會隨迭代數增加**，對 1000 代以上的真實工作負載可忽略。穩態每代 GPU 工作量約 **169 µs**（與程式回報的 AvgIter≈0.185ms 吻合）。

### 3. Kernel 熱點（kernel_stats）— **這才是優化重點**

| Kernel | 佔比 | 平均/次 | 呼叫 |
|---|---|---|---|
| **`qubo_energy_kernel_optimized<256>`** | **80.15%** | **135.7 µs** | 1001 |
| rocprim radix_sort (能量排序) | 8.78% | 14.9 µs | 1001 |
| `updateQ_kernel` | 5.26% | 8.9 µs | 1000 |
| `generate_neighbours_kernel` | 3.97% | 6.7 µs | 1001 |
| `update_global_best_kernel` | 0.95% | 1.6 µs | 1001 |
| `init_sequence_kernel` | 0.87% | 1.5 µs | 1001 |

**能量計算 kernel 一個就吃掉 80% 的 GPU 時間，比其他所有 kernel 加起來還多 4 倍。** 任何優化都應該集中在這裡。

---

### 4. 為什麼 `qubo_energy_kernel` 這麼慢 — 原始碼分析

`src/hip/kernels.hpp:48-80`：

```cpp
int nbr = blockIdx.x;                       // 1 block / neighbour → grid 只有 N=50 個 block
for (int i = tid; i < n_items; i += 256)    // 外層分給 256 threads
    if (x[i])
        for (int j = 0; j < n_items; ++j)   // 內層 O(n) 全掃
            if (x[j]) thread_sum += Qi[j];   // Q 是 double
```

三個結構性問題：

1. **FP64 在消費級 RDNA4 上極慢**（最大嫌疑）。`Q` 與累加都是 `double`。RX 9070 XT 是遊戲卡，FP64 吞吐通常只有 FP32 的 **1/16~1/32**。500×500 的 QUBO 雙重迴圈用 double，正好打在這張卡最弱的點上。
2. **Grid 佔用率過低**：grid = N = **50 個 block**，但這張卡有 ~64 個 CU。整個 GPU 有近一半閒置，且無法靠多 block 隱藏延遲。
3. **演算法 O(n²) + 分支**：每個 neighbour 做 250k 次 `Qi[j]` 讀取，內層 `if(x[j])` 造成 warp divergence 與不規則記憶體存取。

### 5. 建議（依投報率排序）

1. **把 `Q` 與累加改成 `float`**（若精度允許）。這最可能帶來數倍加速，因為避開了 RDNA4 的 FP64 懲罰。可先做數值驗證確認 energy 排序結果不變。→ 預期吃掉 80% 的那塊大幅縮短。
2. **預先壓縮 set-bit 索引**：先把 `x` 中為 1 的 index 收集到 shared memory 的小陣列，內層只迭代這些 index，消除 `if(x[j])` 分支與無效讀取（從 O(n²) 降到 O(k²)，k=被選中的物品數）。
3. **提高並行度**：目前 1 block/neighbour。可讓多個 block 協作一個 neighbour（再 atomic/二階段 reduce），或一次處理整個 population 提高 CU 佔用。
4. 其餘 kernel（sort 8.8%、updateQ 5.3%）目前不值得碰，先把 80% 那塊解決。

---

**原始 CSV 在 `prof/`**，可用 `--output-format pftrace` 重跑後丟進 Perfetto (ui.perfetto.dev) 看時間軸。要我直接動手把 energy kernel 改成 FP32 版本並做數值對照測試嗎？
