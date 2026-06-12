# AEQTS + QUBO (CUDA/HIP Version)

本專案為 **AEQTS** 結合 **QUBO** 模型的 CUDA/HIP 平行化實作。主要用於解決背包問題 (Knapsack Problem)。

程式碼透過 NVIDIA/AMD GPU 進行大規模平行運算，加速量子位元 (Qubit) 的觀測 (Measure)、能量評估 (Energy Evaluation) 以及旋轉閘 (Rotation Gate) 的更新。

## 功能特點

* **QUBO 模型化**：採用 Teacher formulation (固定懲罰係數 ) 將背包問題轉換為 QUBO 矩陣。
* **CUDA 平行加速**：
* **觀測 (Measure)**：使用 `curand` 並行生成隨機數，根據  機率坍縮量子態。
* **能量評估**：平行計算  系統能量。
* **量子閘更新**：基於種群中「最佳」與「最差」解的差異，並行更新每個量子位元的旋轉角度 ()。

* **CUB 整合**：利用 `cub::DeviceRadixSort::SortPairs` 在 GPU 端快速對能量進行排序，以篩選菁英個體。

## 系統需求

* **作業系統**：Linux (建議 Fedora 或 Ubuntu)
* **硬體**：NVIDIA/AMD GPU (需支援 CUDA/HIP)
* **軟體環境**：
* NVIDIA CUDA Toolkit (包含 `nvcc` 編譯器與 `cub` 函式庫)
* AMD ROCm (包含 `hipcc` 編譯器與 `hipcub` 函式庫)
* 支援 C++17 或更高標準的編譯器

## 編譯方式

### 使用 CUDA 後端

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DBACKEND=CUDA
cmake --build build --config Release
```

### 使用 HIP 後端

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DBACKEND=HIP
cmake --build build --config Release
```

## 使用方法

程式執行時可接受命令列參數以指定亂數種子 (Seed)。

### 語法

```bash
./aeqts_qubo [SEED]
./aeqts_qubo --seed [SEED]
```

### 範例

```bash
./aeqts_qubo

./aeqts_qubo 9999
```

## 演算法參數與邏輯

程式內部預設參數如下（可於原始碼 `main` 函式中調整）：

* **Items ()**：500 個物品。
* **Population ()**：50 個個體 (量子染色體)。
* **Iterations**：1000 代。
* **Penalty ()**：10.0 (用於懲罰超出背包負重的解)。

### 核心流程 (Kernel)

1. **初始化 (`init_curand_states`)**：配置每個執行緒獨立的 `curand` 狀態。
2. **主迴圈**：
   * **生成鄰域 (`generate_neighbours_kernel`)**：根據量子位元的機率幅  進行觀測，產生二進位解 ()。
   * **計算能量 (`qubo_energy_kernel`)**：計算目標函數值 。
   * **排序 (`cub::DeviceRadixSort::SortPairs`)**：依據能量由小到大排序，能量越低代表解越佳。
   * **更新 Q-bit (`updateQ_kernel`)**：
   * 選取前  個最佳解與後  個最差解配對。
   * 計算差異向量並套用動態旋轉角度 。
   * 更新  數值。

## 輸出範例

程式將執行 10 次獨立實驗 (Experiment)，並輸出每次執行的最佳能量、背包總價值 (Val)、總重量 (W) 以及平均迭代時間。

```text
Run 1: Energy=-1234.5 | Val=850.0 | W=245.0/250.0 | VALID | AvgIter=0.45 ms
Run 2: Energy=-1240.2 | Val=855.0 | W=248.0/250.0 | VALID | AvgIter=0.44 ms
...

```

## 注意事項

* **記憶體管理**：程式已包含完整的 `cudaMalloc` 與 `cudaFree` 流程，但在修改規模 ( 或 ) 時，請留意 GPU 記憶體 (VRAM) 的使用量。
* **CUDA/HIP 錯誤處理**：程式內建 `CUDA_CHECK` 巨集，若發生執行期錯誤 (如 Kernel launch failure)，將會輸出錯誤訊息並終止程式。

## 授權

禁止散佈給第三方，僅供個人學術研究使用。
