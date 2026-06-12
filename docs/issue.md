# 問題清單 (Issue List)

> 建立日期：2026-06-13
> 由程式碼檢查整理,依嚴重度排序,逐項修正。

## 嚴重 (Blocker)

### #1 「multi node」尚未實作,commit 與實際狀態不符
- 最新 commit `17cbe66 feat: done multi node impl`,但 `src/main.cu` 完全沒有 MPI 程式碼(`grep MPI_ src/` → 0 筆)。
- 影響:目前只是把同一份工作丟到多個 GPU,並非真正多節點分散運算。
- 相關:#2 #3 #4

### #2 CMake 有 MPI 配線但原始碼沒使用
- `CMakeLists.txt` 有 `ENABLE_MPI` 選項與 `USE_MPI` compile definition,
  但原始碼裡沒有任何 `#ifdef USE_MPI`,定義出來也不會被使用。
- 需求:在 `src/main.cu` 加入以 `USE_MPI` 包覆的 MPI 初始化、工作分配與結果彙整邏輯。

### #3 各 rank 用相同種子,結果重複且零通訊
- `script/slurm/nano4/run_mpi.slurm` 用 `mpirun` 啟動 2 節點 × 8 GPU = 16 個 rank,
  但每個 rank 都沒帶 `--run_id`,全部用同一種子 `12345` 跑出完全相同結果。
- 需求:依 MPI rank 指定 seed/run_id,並用 `MPI_Allreduce` 彙整 global best。
- 另需:每個 rank 綁定對應的 GPU(`cudaSetDevice(local_rank)`)。

### #4 Slurm 建置腳本沒開 MPI
- `script/slurm/nano4/mpi.slurm`、`build_mpi.slurm` 的 cmake 沒傳 `-DENABLE_MPI=ON`,
  會在 MPI 關閉狀態下建置。
- 需求:加入 `-DENABLE_MPI=ON`。

## 次要 (Minor)

### #5 README 排序方法敘述過時
- README 寫 `thrust::sort_by_key`,實際程式碼使用 `cub::DeviceRadixSort::SortPairs`。
- 需求:更新 README 對應段落。

### #6 縮排跑掉
- `src/main.cu:110`(`std::vector<unsigned char> best_sol_h`)、
  `src/main.cu:193`(`double final_global_best_energy`)等處貼齊左側。
- 需求:套用 clang-format。

## 修正進度

- [x] #1 / #2 / #3 / #4 — MPI 多節點實作(**計畫 A 島嶼模型**;B 見 `docs/plan-b-population-split.md`)
  - `src/main.cu`:`#ifdef USE_MPI` 包覆 → `MPI_Init`、`MPI_Comm_split_type` 取 local rank
    並 `cudaSetDevice` 綁定 GPU;seed 納入 `world_rank`;結尾 `MPI_Allreduce(MPI_MINLOC)`
    挑全域最佳 rank,`MPI_Bcast` 廣播其解,rank 0 輸出(含 `BestRank`)。
  - `script/slurm/nano4/{mpi,build_mpi}.slurm`:cmake 加 `-DENABLE_MPI=ON`。
- [x] #5 — README 更新(`thrust::sort_by_key` → `cub::DeviceRadixSort::SortPairs`)
- [x] #6 — clang-format(修正 `main.cu` 左側貼齊等)

## 待驗證
- 本機無 `nvcc`,尚未實際編譯。需在叢集上以
  `cmake -S . -B build -DBACKEND=CUDA -DENABLE_MPI=ON` 建置 + `mpirun` 驗證:
  - 各 rank 確實綁到不同 GPU(`cudaSetDevice`)。
  - 多 rank 輸出單一彙整結果且 `BestRank` 合理。
  - 單機(不開 MPI)建置仍正常。
