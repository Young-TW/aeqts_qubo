#!/usr/bin/env bash
# 用 rocprofv3 採集硬體效能計數器，補足 kernel-trace 看不到的「為什麼慢」。
#
# 重要前提（實測於本機 gfx1201 / RX 9070 XT / ROCm 7.13）：
#   * omniperf(rocprof-compute) 不支援 gfx1201（無 SoC 設定檔），裝了也跑不了。
#   * 更關鍵：這張 RDNA4 消費卡在此 ROCm 版本只開放「週期級」計數器。
#     指令級(SQ_INSTS_VALU/SALU)、記憶體/快取(TA/TCP/TCC/GL2C) 一律回 0，
#     所以 VALU 飽和度、記憶體頻寬、L2 命中率、roofline 都「拿不到資料」。
#   實測唯一有值的 4 個 counter：
#     GRBM_COUNT、GRBM_GUI_ACTIVE、SQ_BUSY_CYCLES、SQ_WAVES
#   本腳本只收這組（單一 pass，無 replay），不收一堆會回 0 的指標誤導判讀。
#   要看 compute/memory 拆分，目前只能靠 kernel-trace 計時 + 原始碼推理
#   (見 docs/rocprofv3.md)。
#
# 用法:
#   scripts/profile_counters.sh                 # 預設只測 energy kernel
#   scripts/profile_counters.sh ".*"            # 測全部 kernel
#   KERNEL=updateQ ITER=200 SEED=42 scripts/profile_counters.sh
#
# 環境變數:
#   KERNEL  kernel 名稱的 regex (預設 qubo_energy)；亦可用第一個位置參數
#   ITER    迭代代數 (預設 200)
#   SEED    亂數種子 (預設 12345，與既有報告一致)
#   BIN     執行檔 (預設 ./build/aeqts_qubo)
#   OUTDIR  輸出目錄 (預設 prof/counters_<kernel>)
set -euo pipefail
cd "$(dirname "$0")/.."

KERNEL="${1:-${KERNEL:-qubo_energy}}"
ITER="${ITER:-200}"
SEED="${SEED:-12345}"
BIN="${BIN:-./build/aeqts_qubo}"
OUTDIR="${OUTDIR:-prof/counters_$(echo "$KERNEL" | tr -c 'A-Za-z0-9' '_' | sed 's/_*$//')}"

command -v rocprofv3 >/dev/null || { echo "找不到 rocprofv3，請先 source /opt/rocm 環境。" >&2; exit 1; }
[ -x "$BIN" ] || { echo "找不到執行檔 $BIN，請先 cmake --build build。" >&2; exit 1; }

rm -rf "$OUTDIR"
echo ">> kernel=/$KERNEL/  iter=$ITER  seed=$SEED  ->  $OUTDIR"

rocprofv3 \
  --kernel-include-regex "$KERNEL" \
  --pmc GRBM_GUI_ACTIVE GRBM_COUNT SQ_BUSY_CYCLES SQ_WAVES \
  --output-format csv -d "$OUTDIR" -o counters \
  -- "$BIN" --iter "$ITER" "$SEED" >/dev/null

echo
python3 scripts/summarize_counters.py "$OUTDIR"
echo
echo "原始 per-dispatch CSV：$OUTDIR/counters_counter_collection.csv"
