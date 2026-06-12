# 問題清單 (Issue List)

> 建立日期：2026-06-13(2026-06-13 更新:#1–#8 已修正並移除,#9 待討論)
> 由程式碼檢查整理,依嚴重度排序,逐項修正。

## 次要 (Minor)

### #9 `config/case.toml` 並非合法 TOML,副檔名易誤導
- 檔案以 `.toml` 命名,但內容用 `key = value;`(行尾分號)與 `//` 註解,
  這些都不是合法 TOML 語法;`include/config.h` 的 `load_config` 也是自製的
  `key = value` 解析器,不是 TOML parser。
- 影響:之後若有人改用真正的 TOML 函式庫讀取會直接失敗;命名與實作不一致。
- 需求:二擇一 —— 改名為 `case.conf` / `case.ini` 等以反映實際格式,
  或改用合法 TOML 並換成真正的 TOML 解析器。
- 附帶:`include/config.h:9` 註解「預設值與原本 main.cu 內建值相同」仍指向已不存在的
  `main.cu`(已重構為 `main.cpp`),順手更新。

## 待驗證
- 本機無 `nvcc`,尚未實際編譯。需在叢集上以
  `cmake -S . -B build -DBACKEND=CUDA -DENABLE_MPI=ON` 建置 + `mpirun` 驗證:
  - 各 rank 確實綁到不同 GPU(`gpu_set_device(local_rank)` → `cudaSetDevice`)。
  - 多 rank 輸出單一彙整結果且 `BestRank` 合理。
  - 單機(不開 MPI)建置仍正常。
