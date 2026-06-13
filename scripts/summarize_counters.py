#!/usr/bin/env python3
"""把 profile_counters.sh 採集的計數器 CSV 彙整成一張可讀的表。

對每個 kernel 取所有 dispatch 的平均，並由「週期級」counter 推導出本卡
(gfx1201) 唯一拿得到的硬體訊號：並行度(waves/CU) 與 SQ 工作密度。

說明：RX 9070 XT(RDNA4) 在 ROCm 7.13 只開放週期級 counter，VALU/記憶體/
快取一律回 0，因此無法判讀 compute-bound vs memory-bound — 那部分請看
kernel-trace 計時與原始碼分析(docs/rocprofv3.md)。

用法: summarize_counters.py <outdir>
依賴: 只用標準函式庫(配合 scripts/fold_to_flamegraph.py 的 no-deps 風格)。
"""
import csv
import glob
import os
import sys
from collections import defaultdict


def short_kernel(name: str) -> str:
    # "void qubo_energy_kernel_optimized<128>(...)" -> "qubo_energy_kernel_optimized<128>"
    return name.split("(", 1)[0].replace("void ", "").strip()


def find_csvs(outdir: str):
    # 單一 pass 直接落在 outdir；multi-pass 才有 pass_*/ 子目錄。
    files = glob.glob(os.path.join(outdir, "*counter_collection.csv"))
    files += glob.glob(os.path.join(outdir, "pass_*", "*counter_collection.csv"))
    return sorted(set(files))


def read_gpu_info(outdir: str):
    """回傳 (CU 數, wavefront 大小)；只取 GPU agent，避開 CPU agent 那列。"""
    for f in glob.glob(os.path.join(outdir, "**", "*agent_info.csv"), recursive=True):
        for row in csv.DictReader(open(f)):
            if row.get("Agent_Type") == "GPU":
                cu = int(row.get("Cu_Count") or 0)
                ws = int(row.get("Wave_Front_Size") or 0)
                return cu, ws
    return 0, 0


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 1
    outdir = sys.argv[1]
    files = find_csvs(outdir)
    if not files:
        print(f"在 {outdir} 找不到 counter_collection.csv", file=sys.stderr)
        return 1
    cu, wave_size = read_gpu_info(outdir)

    # kernel -> counter -> [values]，並記錄每個 dispatch 的 grid size
    data = defaultdict(lambda: defaultdict(list))
    grids = defaultdict(list)
    for f in files:
        for row in csv.DictReader(open(f)):
            k = short_kernel(row["Kernel_Name"])
            try:
                data[k][row["Counter_Name"]].append(float(row["Counter_Value"]))
            except ValueError:
                continue
            g = row.get("Grid_Size")
            if g and g.isdigit():
                grids[k].append(int(g))

    for kernel in sorted(data):
        c = data[kernel]
        ndisp = max((len(v) for v in c.values()), default=0)
        mean = {name: sum(v) / len(v) for name, v in c.items() if v}
        print(f"━━ {kernel}  ({ndisp} 次 dispatch 平均) ━━")

        active = mean.get("GRBM_GUI_ACTIVE")
        total = mean.get("GRBM_COUNT")
        sqbusy = mean.get("SQ_BUSY_CYCLES")
        waves = mean.get("SQ_WAVES")

        if active is not None:
            print(f"  GPU 活躍週期 (GRBM_GUI_ACTIVE)   {active:14,.0f}  ∝ kernel 執行時間")
        if total is not None and active is not None and total:
            print(f"  GPU 活躍佔比                    {active / total * 100:13.1f}%  活躍週期 / 總週期")
        if sqbusy is not None:
            print(f"  SQ 忙碌週期 (SQ_BUSY_CYCLES)     {sqbusy:14,.0f}  指令仲裁器工作週期(跨 SIMD 累加)")
        if waves is not None:
            print(f"  SQ_WAVES / dispatch             {waves:14,.0f}  SQ 計到的 wave 數")
        g = sum(grids[kernel]) / len(grids[kernel]) if grids.get(kernel) else None
        if g is not None:
            print(f"  平均 grid size                  {g:14,.0f}  threads")
            if wave_size:
                print(f"  grid 推算 wave 數               {g / wave_size:14,.0f}  grid / wavefront({wave_size})")
        print("  " + "─" * 46)
        for line in verdict(g, wave_size, cu):
            print(f"  → {line}")
        print()
    return 0


def verdict(grid, wave_size, cu):
    out = []
    if grid is not None and wave_size and cu:
        waves_per_cu = grid / wave_size / cu
        if waves_per_cu < 1:
            out.append(f"grid 僅 {grid:,.0f} threads → 每 CU 不到 1 個 wave"
                       f"(CU={cu})：並行度嚴重不足，整張卡大半閒置。")
        else:
            out.append(f"grid {grid:,.0f} threads ÷ wave{wave_size} ÷ {cu} CU "
                       f"≈ {waves_per_cu:.1f} wave/CU 同時可駐留。")
    out.append("compute-bound vs memory-bound 在 gfx1201 量不到"
               "(VALU/記憶體/快取 counter 全回 0)；")
    out.append("請對照 kernel-trace 計時與原始碼分析 (docs/rocprofv3.md)。")
    return out


if __name__ == "__main__":
    sys.exit(main())
