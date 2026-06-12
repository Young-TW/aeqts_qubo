# 問題清單 (Issue List)

> 建立日期：2026-06-13(2026-06-13 更新:#1–#9 已修正並移除)
> 由程式碼檢查整理,依嚴重度排序,逐項修正。

## 待驗證
- 本機無 `nvcc`,尚未實際編譯。需在叢集上以
  `cmake -S . -B build -DBACKEND=CUDA -DENABLE_MPI=ON` 建置 + `mpirun` 驗證:
  - 各 rank 確實綁到不同 GPU(`gpu_set_device(local_rank)` → `cudaSetDevice`)。
  - 多 rank 輸出單一彙整結果且 `BestRank` 合理。
  - 單機(不開 MPI)建置仍正常。
