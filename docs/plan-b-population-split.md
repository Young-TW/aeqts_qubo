# 計畫 B：族群分割 (Population Split) 多節點實作

> 狀態：**保留待辦**,目前先採用計畫 A(島嶼模型)。
> 本文件保存 B 方案的設計,供日後需要更高平行度時實作。

## 動機
島嶼模型 (A) 每個 rank 各跑完整 AEQTS,通訊極少,但單一個體族群規模 (N) 受限於單 GPU。
當 N 需要放大到單卡裝不下、或想讓所有 rank 共同演化「同一個」族群時,改用族群分割。

## 核心概念
把族群的 N 個個體切成 `world_size` 份,每個 rank 負責 `N_local = N / world_size` 個個體,
每一代結束後做一次全域同步,讓旋轉閘更新看得到全域的最佳/最差解。

## 每代流程
1. 各 rank 對自己的 `N_local` 個個體:`generate_neighbours` → `qubo_energy`。
2. **跨 rank 排序 / 菁英挑選**(關鍵步驟,兩種做法):
   - (b1) 各 rank 算出本地排序後,以 `MPI_Allgather` 收集各 rank 的局部最佳/最差個體
     (能量 + 解向量),在每個 rank 重組出「全域前 N/2 最佳、後 N/2 最差」配對。
   - (b2) 用 `MPI_Allreduce` 搭配 `MPI_MINLOC`/`MPI_MAXLOC` 找全域極值,再廣播對應解向量。
3. 各 rank 用全域配對資訊做 `updateQ`(旋轉閘),維持各自負責的 alpha/beta 區段一致。

## 通訊量估計
- 每代需傳輸的解向量:約 `O(N * n_items)` bytes(`unsigned char`)。
- 以 N=50、n_items=5000 為例,每代約 250 KB × 2(best/worst)≈ 0.5 MB/代。
- iter=5000 → 約 2.5 GB 總通訊;CUDA-aware MPI 直接從 device buffer 傳可省去 H2D/D2H。

## 需要的改動
- `src/main.cu`:迴圈內插入 MPI 集合通訊;alpha/beta 的全域一致性管理。
- 新增 kernel:把本地 best/worst 打包成連續 buffer 以利 `MPI_Allgather`。
- `updateQ_kernel`:輸入改為全域配對索引,而非單一 rank 的 `sorted_idx`。
- GPU 綁定:`cudaSetDevice(local_rank)`(與 A 相同)。

## 與 A 的取捨
| 項目 | A 島嶼模型 | B 族群分割 |
|------|-----------|-----------|
| 通訊頻率 | 僅結尾一次 | 每代一次 |
| 程式改動 | 小 | 大 |
| 有效族群規模 | 單卡上限 × rank 數(獨立) | 可線性擴大(共演化) |
| 收斂行為 | 多重獨立搜尋取最佳 | 單一大族群協同搜尋 |
| 風險 | 低 | 高(同步、負載平衡) |

## 實作前置條件
- A 已驗證可正確跑 CUDA-aware MPI(GPU 綁定、Allreduce 路徑都通)。
- 有 micro-benchmark 確認每代通訊不會吃掉平行加速。
