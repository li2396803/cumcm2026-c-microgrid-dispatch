"""生成 v2-D 策略在题目要求的表 1/表 2/表 3 格式下的结果表 (Markdown)."""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model as M
import pipeline as P

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "out")
DATES4 = ["2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21"]
WIN = [60, 72, 84, 96, 108, 120]
BLK = [("0:00-4:00", 0, 24), ("4:00-8:00", 24, 48), ("8:00-12:00", 48, 72),
       ("12:00-16:00", 72, 96), ("16:00-20:00", 96, 120),
       ("20:00-24:00", 120, 144)]


def lab(k):
    a, b = k * 10, (k + 1) * 10
    fa = f"{a//60}:{a%60:02d}"
    fb = "24:00" if b == 1440 else f"{b//60}:{b%60:02d}"
    return f"{fa}-{fb}"


def segs(e):
    out, k = [], 0
    while k < 144:
        if e[k] > 1e-6:
            k0 = k
            while k < 144 and e[k] > 1e-6:
                k += 1
            out.append((k0, k - 1, float(e[k0:k].sum())))
        else:
            k += 1
    return out


def main():
    data = P.load_data()
    dates = [str(x) for x in data["dates2"]]
    p = data["att1"][:, 0]
    m = np.load(os.path.join(OUT, "v2_mpc_b.npz"), allow_pickle=True)
    dd = [str(x) for x in m["dates"]]
    L = []
    L.append("#### 15.6.1 表 1 格式：微网在指定时间段的购电量（v2-D，单位 kWh）")
    L.append("")
    L.append("| 日期 | " + " | ".join(lab(k) for k in WIN) +
             " | 全天购电量 | 全天购电费(元) |")
    L.append("|" + "---|" * (len(WIN) + 3))
    for nm in DATES4:
        j = dd.index(nm)
        b = m["b"][j]
        L.append("| %s | %s | %.2f | %.2f |" % (
            nm, " | ".join("%.2f" % b[k] for k in WIN),
            b.sum(), float(m["cost"][j])))
    L.append("")
    L.append("#### 15.6.2 表 2 格式：储能设备在指定时间段的充放电量与 0:00/24:00 储电量（v2-D，单位 kWh）")
    L.append("")
    L.append("| 日期 | 时段 | 充电量 | 放电量 | 时段 | 充电量 | 放电量 | 0:00 储电量 | 24:00 储电量 |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for nm in DATES4:
        j = dd.index(nm)
        c, d = m["c"][j], m["d"][j]
        s = 6000.0 + np.cumsum(0.9 * c - d / 0.9)
        for t, ((n1, a1, b1), (n2, a2, b2)) in enumerate(zip(BLK[:3], BLK[3:])):
            L.append("| %s | %s | %.2f | %.2f | %s | %.2f | %.2f | %s | %s |" % (
                nm if t == 0 else "", n1, c[a1:b1].sum(), d[a1:b1].sum(),
                n2, c[a2:b2].sum(), d[a2:b2].sum(),
                "6000.00" if t == 0 else "",
                "%.2f" % s[-1] if t == 0 else ""))
    L.append("")
    L.append("#### 15.6.3 表 3 格式：微网在指定日期的紧急购电量（v2-D，单位 kWh）")
    L.append("")
    L.append("| 日期 | 紧急购电时段与电量 | 合计 | 紧急购电费(元) |")
    L.append("|---|---|---|---|")
    for nm in DATES4:
        j = dd.index(nm)
        e = m["emergency"][j]
        S = segs(e)
        txt = "；".join("%s—%s: %.1f" % (lab(a).split("-")[0], lab(b).split("-")[1], v)
                        for a, b, v in S) or "无"
        cost = float(np.sum(5.0 * p * e))
        L.append("| %s | %s | %.1f | %.0f |" % (nm, txt, e.sum(), cost))
    txt = "\n".join(L)
    open(os.path.join(OUT, "v2_tables_final.md"), "w", encoding="utf-8").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
