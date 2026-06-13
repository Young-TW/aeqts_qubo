#!/usr/bin/env python3
"""跨參數區間的效能 / 正確性驗證harness。

用途:判斷一項 kernel 最佳化是否「通用」——也就是在不同 (n_items, iter, N)
都不會回歸。只有在所有 case 都 >= baseline(或在量測雜訊內)才算通過。

設計要點(對應 docs/performance.md 與記憶中的量測陷阱):
  * GPU 會在閒置時降頻(mclk 掉到 ~772 MHz),冷啟動也含 lazy context 開銷。
    -> 每個 case 先做 warmup 再正式量,且把多個 binary「交錯」跑(round-robin),
       讓兩版輪流吃到同樣的時脈狀態,A/B 才公平。
  * 單次 run 有抖動 -> 每個 case 重複 --reps 輪,取「中位數」而非平均。
  * 正確性:以第一個 binary 當參考,比較各 case 回報的 Energy 相對誤差與 VALID。
    (FP32 路徑下能量到 6 位有效數字應一致;只看是否在 --etol 內。)

範例:
  # 純量基準(單一 binary),掃多個尺寸,把結果存成 baseline.json
  scripts/validate_perf.py --bin build/aeqts_qubo:baseline --save baseline.json

  # A/B:baseline vs 候選版,交錯量測、判斷是否通用
  scripts/validate_perf.py \
      --bin build_base/aeqts_qubo:base \
      --bin build_opt/aeqts_qubo:opt \
      --reps 3

  # 自訂 case(items,iter,N),可重複
  scripts/validate_perf.py --bin build/aeqts_qubo --case 4000,1000,64 --case 1000,1000,50
"""

import argparse
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

ENERGY_RE = re.compile(r"Energy=([-+0-9.eE]+)")
AVGITER_RE = re.compile(r"AvgIter=([0-9.eE+]+)\s*ms")
VALID_RE = re.compile(r"\|\s*(VALID|OVERWEIGHT)\s*\|")

# 預設 case 涵蓋兩種瓶頸區間(見 qubo-energy-kernel-bottleneck 記憶):
#   小 n -> Q 常駐 cache,compute/cache-bound
#   大 n -> Q 超出 cache,VRAM 頻寬 bound (使用者實際 config 為 4000)
# iter 取適中讓每個 case 數秒內跑完;N 用 64(實際 config)。
DEFAULT_CASES = [
    (1000, 1000, 64),
    (1000, 1500, 64),
    (2000, 1000, 64),
    (2000, 2000, 64),
    (3000, 800, 64),
    (4000, 600, 64),
]


def parse_case(s):
    parts = s.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("case 格式須為 items,iter,N")
    return tuple(int(p) for p in parts)


def parse_bin(s):
    if ":" in s:
        path, label = s.rsplit(":", 1)
    else:
        path, label = s, Path(s).parent.name or s
    p = Path(path)
    if not p.is_absolute():
        p = REPO / p
    if not p.exists():
        sys.exit(f"找不到執行檔: {p}")
    return (str(p), label)


def run_one(binary, items, iter_, N, seed):
    """跑一次,回傳 (avg_iter_ms, energy, valid)。"""
    cmd = [
        binary,
        "--items",
        str(items),
        "--iter",
        str(iter_),
        "--N",
        str(N),
        "--seed",
        str(seed),
    ]
    out = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True)
    if out.returncode != 0:
        sys.exit(f"執行失敗 ({binary}):\n{out.stderr}\n{out.stdout}")
    text = out.stdout
    m_t = AVGITER_RE.search(text)
    m_e = ENERGY_RE.search(text)
    m_v = VALID_RE.search(text)
    if not (m_t and m_e):
        sys.exit(f"無法解析輸出 ({binary}):\n{text}")
    return (
        float(m_t.group(1)),
        float(m_e.group(1)),
        (m_v.group(1) if m_v else "?"),
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--bin",
        action="append",
        type=parse_bin,
        required=True,
        metavar="PATH[:LABEL]",
        help="待測執行檔,可重複;第一個為正確性/速度的參考基準。",
    )
    ap.add_argument(
        "--case",
        action="append",
        type=parse_case,
        metavar="ITEMS,ITER,N",
        help="自訂 case,可重複。",
    )
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--reps", type=int, default=3, help="每個 case 交錯量測的輪數。")
    ap.add_argument("--warmup", type=int, default=1, help="正式量測前的暖機輪數。")
    ap.add_argument(
        "--etol",
        type=float,
        default=1e-4,
        help="能量相對誤差容忍(超過視為結果不一致)。",
    )
    ap.add_argument(
        "--rtol",
        type=float,
        default=0.02,
        help="判定回歸的時間相對門檻;慢於基準超過此比例才算回歸。",
    )
    ap.add_argument("--save", metavar="JSON", help="把結果存成 JSON。")
    args = ap.parse_args()

    bins = args.bin
    cases = args.case if args.case else DEFAULT_CASES
    ref_label = bins[0][1]

    print(
        f"參考基準: {ref_label}   seed={args.seed}   reps={args.reps}   "
        f"warmup={args.warmup}"
    )
    print(f"binaries: " + ", ".join(f"{l}={p}" for p, l in bins))
    print()

    results = []  # 每筆: dict(case, per-bin timings/energy)
    regressions = []
    mismatches = []

    for items, iter_, N in cases:
        # 暖機(把時脈拉起來、付掉冷啟動)
        for _ in range(args.warmup):
            for path, _label in bins:
                run_one(path, items, iter_, N, args.seed)

        # 交錯量測:每輪每個 binary 各跑一次
        times = {label: [] for _, label in bins}
        energy = {}
        valid = {}
        for _ in range(args.reps):
            for path, label in bins:
                t, e, v = run_one(path, items, iter_, N, args.seed)
                times[label].append(t)
                energy[label] = e
                valid[label] = v

        med = {label: statistics.median(ts) for label, ts in times.items()}
        ref_t = med[ref_label]
        ref_e = energy[ref_label]

        row = {
            "items": items,
            "iter": iter_,
            "N": N,
            "median_ms": med,
            "energy": energy,
            "valid": valid,
        }
        results.append(row)

        print(f"=== items={items}  iter={iter_}  N={N} ===")
        hdr = f"  {'binary':<12} {'med ms':>10} {'speedup':>9} {'energy':>14} {'valid':>10}"
        print(hdr)
        for _path, label in bins:
            sp = ref_t / med[label] if med[label] > 0 else float("nan")
            ediff = abs(energy[label] - ref_e) / max(abs(ref_e), 1e-30)
            flag = ""
            if label != ref_label:
                if med[label] > ref_t * (1 + args.rtol):
                    flag = "  <== REGRESSION"
                    regressions.append((items, iter_, N, label, sp))
                if ediff > args.etol:
                    flag += f"  <== ENERGY DIFF {ediff:.2e}"
                    mismatches.append((items, iter_, N, label, ediff))
            print(
                f"  {label:<12} {med[label]:>10.4f} {sp:>8.3f}x "
                f"{energy[label]:>14.6g} {valid[label]:>10}{flag}"
            )
        print()

    # 總結
    print("================ 總結 ================")
    if len(bins) == 1:
        print("(單一 binary:僅記錄基準,無 A/B 判定)")
    else:
        if regressions:
            print(f"❌ 發現 {len(regressions)} 個回歸 case(慢於基準 >{args.rtol:.0%}):")
            for items, iter_, N, label, sp in regressions:
                print(f"   - {label} @ items={items},iter={iter_},N={N}: {sp:.3f}x")
        else:
            print(f"✅ 無回歸:所有候選在每個 case 都 >= 基準(容忍 {args.rtol:.0%})。")
        if mismatches:
            print(f"⚠️  發現 {len(mismatches)} 個能量不一致(>{args.etol:.0e}):")
            for items, iter_, N, label, ediff in mismatches:
                print(
                    f"   - {label} @ items={items},iter={iter_},N={N}: 相對差 {ediff:.2e}"
                )
        else:
            print(f"✅ 能量一致:所有候選與基準在 {args.etol:.0e} 內。")

    if args.save:
        out_path = Path(args.save)
        if not out_path.is_absolute():
            out_path = REPO / out_path
        out_path.write_text(
            json.dumps(
                {"seed": args.seed, "reps": args.reps, "results": results},
                indent=2,
                ensure_ascii=False,
            )
        )
        print(f"\n已存檔: {out_path}")

    # 退出碼:有回歸或能量不一致 -> 非 0,方便 CI / 腳本判斷
    return 1 if (regressions or mismatches) else 0


if __name__ == "__main__":
    sys.exit(main())
