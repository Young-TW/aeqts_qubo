#!/usr/bin/env python3
"""Render a folded-stack file into a flame-graph SVG (self-contained, no deps).

Folded format (one line per stack): "frame1;frame2;frame3 <count>"
Width of each box is proportional to its aggregated count; depth grows upward.
"""
import sys, html, colorsys, hashlib

def parse(path):
    root = {"name": "root", "value": 0, "children": {}}
    for line in open(path):
        line = line.rstrip("\n")
        if not line:
            continue
        stack, _, cnt = line.rpartition(" ")
        cnt = int(cnt)
        node = root
        root["value"] += cnt
        for frame in stack.split(";"):
            node = node["children"].setdefault(
                frame, {"name": frame, "value": 0, "children": {}})
            node["value"] += cnt
    return root

def color(name):
    # warm "hot" palette, hue deterministic from name so reruns are stable
    h = int(hashlib.md5(name.encode()).hexdigest(), 16)
    hue = (h % 60) / 360.0            # 0..60deg -> red..yellow
    r, g, b = colorsys.hsv_to_rgb(hue, 0.55 + (h % 30) / 100.0, 0.95)
    return f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"

def main():
    src, out = sys.argv[1], sys.argv[2]
    title = sys.argv[3] if len(sys.argv) > 3 else "Flame Graph"
    subtitle = sys.argv[4] if len(sys.argv) > 4 else ""
    root = parse(src)
    total = root["value"]

    W, ROW, PADX, TOP = 1400, 18, 10, 54
    px = (W - 2 * PADX) / total

    rects = []
    maxdepth = [0]

    def walk(node, depth, x0):
        # draw children sorted by name for stable left-to-right order
        cx = x0
        for child in sorted(node["children"].values(), key=lambda n: n["name"]):
            w = child["value"] * px
            if w >= 0.1:
                maxdepth[0] = max(maxdepth[0], depth)
                rects.append((cx, depth, w, child["name"], child["value"]))
                walk(child, depth + 1, cx)
            cx += w

    walk(root, 0, PADX)
    height = TOP + (maxdepth[0] + 1) * ROW + 20

    def y_of(depth):
        return height - 20 - (depth + 1) * ROW

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{height}" '
        f'font-family="Verdana,Helvetica,sans-serif" font-size="12">',
        f'<rect width="{W}" height="{height}" fill="#f8f8f8"/>',
        f'<text x="{W/2}" y="24" text-anchor="middle" font-size="17" '
        f'font-weight="bold">{html.escape(title)}</text>',
    ]
    if subtitle:
        parts.append(f'<text x="{W/2}" y="42" text-anchor="middle" '
                     f'font-size="11" fill="#555">{html.escape(subtitle)}</text>')

    for x, depth, w, name, val in rects:
        y = y_of(depth)
        pct = 100.0 * val / total
        tip = f"{name}  ({val:,} ns, {pct:.2f}%)"
        label = ""
        if w > 28:
            chars = int((w - 6) / 6.5)
            label = name if len(name) <= chars else name[:max(0, chars - 1)] + "…"
        parts.append(
            f'<g><title>{html.escape(tip)}</title>'
            f'<rect x="{x:.1f}" y="{y}" width="{max(w-1,0.5):.1f}" height="{ROW-1}" '
            f'fill="{color(name)}" stroke="#fff" stroke-width="0.5" rx="1"/>'
            f'<text x="{x+3:.1f}" y="{y+ROW-5}">{html.escape(label)}</text></g>')

    parts.append("</svg>")
    open(out, "w").write("\n".join(parts))
    print(f"wrote {out}  ({total:,} ns total, {len(rects)} boxes)")

if __name__ == "__main__":
    main()
