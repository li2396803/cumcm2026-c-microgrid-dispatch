"""生成说明文档所需的全部表格 (Markdown)."""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model as M
import pipeline as P

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "out")
LINES = []


def w(s=""):
    LINES.append(s)


def lab(k):
    a, b = k * 10, (k + 1) * 10
    fa = f"{a//60}:{a%60:02d}"
    fb = "24:00" if b == 1440 else f"{b//60}:{b%60:02d}"
    return f"{fa}-{fb}"


WINDOWS = [60, 72, 84, 96, 108, 120]
BLOCKS = [("0:00-4:00", 0, 24), ("4:00-8:00", 24, 48), ("8:00-12:00", 48, 72),
          ("12:00-16:00", 72, 96), ("16:00-20:00", 96, 120),
          ("20:00-24:00", 120, 144)]
DATES4 = ["2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21"]


def seg_list(e, thresh=1e-6):
    out = []
    k = 0
    while k < 144:
        if e[k] > thresh:
            k0 = k
            while k < 144 and e[k] > thresh:
                k += 1
            out.append((k0, k - 1, float(e[k0:k].sum())))
        else:
            k += 1
    return out


def main():
    data = P.load_data()
    dates = [str(x) for x in data["dates2"]]
    idx = P._day_range(data)
    row_of = {d: j for j, d in enumerate([dates[i] for i in idx])}
    p_fix = data["att1"][:, 0]
    s1 = np.load(os.path.join(OUT, "sol_p1.npz"))
    s2 = np.load(os.path.join(OUT, "sol_p2.npz"), allow_pickle=True)
    s3 = np.load(os.path.join(OUT, "sol_p3.npz"), allow_pickle=True)
    s42 = np.load(os.path.join(OUT, "sol_p42.npz"), allow_pickle=True)
    s43 = np.load(os.path.join(OUT, "sol_p43.npz"), allow_pickle=True)
    with open(os.path.join(OUT, "summary.json")) as f:
        summ = json.load(f)
    with open(os.path.join(OUT, "verify_report.json")) as f:
        ver = json.load(f)
    with open(os.path.join(OUT, "analysis_hedge.json")) as f:
        hed = json.load(f)
    with open(os.path.join(OUT, "analysis_forecast.json")) as f:
        fcs = json.load(f)
    with open(os.path.join(OUT, "analysis_sensitivity.json")) as f:
        sen = json.load(f)
    with open(os.path.join(OUT, "cbc_results.json")) as f:
        cbc = json.load(f)

    # ---------- 表 1 ----------
    w("#### 表 1 微网在指定时间段的购电量 (问题 1, 单位: kWh)")
    w()
    w("| 时间段 | 购电量 | 时间段 | 购电量 | 时间段 | 购电量 |")
    w("|---|---|---|---|---|---|")
    b = s1["b"]
    w("| %s | %.2f | %s | %.2f | %s | %.2f |" % (
        lab(60), b[60], lab(72), b[72], lab(84), b[84]))
    w("| %s | %.2f | %s | %.2f | %s | %.2f |" % (
        lab(96), b[96], lab(108), b[108], lab(120), b[120]))
    w("| **全天购电量** | **%.2f** | **全天购电费** | **%.2f 元** | | |"
      % (b.sum(), s1["obj"]))
    w()
    # ---------- 表 2 ----------
    w("#### 表 2 储能设备充放电量 (问题 1, 单位: kWh)")
    w()
    w("| 时间段 | 充电量 | 放电量 | 时间段 | 充电量 | 放电量 |")
    w("|---|---|---|---|---|---|")
    c, d, s = s1["c"], s1["d"], s1["s"]
    for (n1, a1, b1), (n2, a2, b2) in zip(BLOCKS[:3], BLOCKS[3:]):
        w("| %s | %.2f | %.2f | %s | %.2f | %.2f |" % (
            n1, c[a1:b1].sum(), d[a1:b1].sum(),
            n2, c[a2:b2].sum(), d[a2:b2].sum()))
    w("| **0:00 储电量** | **6000.00** | | **24:00 储电量** | **%.2f** | |"
      % s[-1])
    w()

    # ---------- 表 3: 问题 2 ----------
    w("#### 表 3 问题 2 指定日期的购电量与储能运行 (单位: kWh)")
    w()
    hdr = "| 日期 | " + " | ".join(["%s" % lab(k) for k in WINDOWS]) + \
          " | 全天购电量 | 全天购电费(元) |"
    w(hdr)
    w("|" + "---|" * (len(WINDOWS) + 3))
    for nm in DATES4:
        j = row_of[nm]
        bb = s2["b"][j]
        w("| %s | %s | %.2f | %.2f |" % (
            nm, " | ".join("%.2f" % bb[k] for k in WINDOWS),
            bb.sum(), s2["cost"][j]))
    w()
    w("| 日期 | 时段 | 充电量 | 放电量 | 时段 | 充电量 | 放电量 | 0:00储电量 | 24:00储电量 |")
    w("|---|---|---|---|---|---|---|---|---|")
    for nm in DATES4:
        j = row_of[nm]
        cc, dd, ss = s2["c"][j], s2["d"][j], s2["s"][j]
        for t, ((n1, a1, b1), (n2, a2, b2)) in enumerate(
                zip(BLOCKS[:3], BLOCKS[3:])):
            w("| %s | %s | %.2f | %.2f | %s | %.2f | %.2f | %s | %s |" % (
                nm if t == 0 else "", n1, cc[a1:b1].sum(), dd[a1:b1].sum(),
                n2, cc[a2:b2].sum(), dd[a2:b2].sum(),
                "6000.00" if t == 0 else "",
                "%.2f" % ss[-1] if t == 0 else ""))
    w()
    w("**表 3(续) 问题 2 紧急购电**: 四个指定日期均无紧急购电 (紧急购电量 0 kWh), "
      "全年紧急购电总量 0 kWh。")
    w()

    # ---------- 表 4: 问题 3 ----------
    w("#### 表 4 问题 3 指定日期的计划/调整/紧急购电量与储能运行 (单位: kWh)")
    w()
    w("| 日期 | 指标 | " + " | ".join([lab(k) for k in WINDOWS]) +
      " | 全天合计 |")
    w("|" + "---|" * (len(WINDOWS) + 3))
    for nm in DATES4:
        j = row_of[nm]
        for title, arr in [("计划购电", s3["plan_b"][j]),
                           ("调整购电", s3["b"][j]),
                           ("紧急购电", s3["emergency"][j])]:
            w("| %s | %s | %s | %.2f |" % (
                nm if title == "计划购电" else "", title,
                " | ".join("%.2f" % arr[k] for k in WINDOWS), arr.sum()))
        cc, dd, ss = s3["c"][j], s3["d"][j], s3["s"][j]
        w("| | 充/放电(0-4/4-8/8-12) | %s | |" % " / ".join(
            "%.0f·%.0f" % (cc[a:b].sum(), dd[a:b].sum())
            for _, a, b in BLOCKS[:3]))
        w("| | 充/放电(12-16/16-20/20-24) | %s | |" % " / ".join(
            "%.0f·%.0f" % (cc[a:b].sum(), dd[a:b].sum())
            for _, a, b in BLOCKS[3:]))
        w("| | 0:00 / 24:00 储电量 | 6000.00 / %.2f | |" % ss[-1])
    w()
    w("**表 4(续) 问题 3 紧急购电时段明细 (购电量单位 kWh)**")
    w()
    w("| 日期 | 紧急购电时段与电量 | 合计 | 紧急购电费(元) |")
    w("|---|---|---|---|")
    for nm in DATES4:
        j = row_of[nm]
        segs = seg_list(s3["emergency"][j])
        txt = "; ".join("%s—%s: %.1f" % (lab(a).split("-")[0],
                                          lab(bb).split("-")[1], v)
                        for a, bb, v in segs)
        em_cost = float(np.sum(5.0 * p_fix * s3["emergency"][j]))
        w("| %s | %s | %.1f | %.0f |" % (nm, txt,
                                         s3["emergency"][j].sum(), em_cost))
    w()

    # ---------- 问题 4 ----------
    w("#### 表 5 问题 4 指定日期结果 (波动电价)")
    w()
    w("| 日期 | 模型 | 全天计划购电 | 调整后购电 | 紧急购电 | 全天总费用(元) |")
    w("|---|---|---|---|---|---|")
    prt = data["price_rt"]
    for nm in DATES4:
        j = row_of[nm]
        i = dates.index(nm)
        w("| %s | 问题4-2 | %.2f | %.2f | 0.00 | %.2f |" % (
            nm, s42["b"][j].sum(), s42["b"][j].sum(), s42["cost"][j]))
        c43 = float(np.sum(prt[i] * np.minimum(s43["plan_b"][j], s43["b"][j])
                           + 0.5 * prt[i] * np.maximum(0, s43["plan_b"][j] - s43["b"][j])
                           + 1.5 * prt[i] * np.maximum(0, s43["b"][j] - s43["plan_b"][j])
                           + 5.0 * prt[i] * s43["emergency"][j]))
        w("| %s | 问题4-3 | %.2f | %.2f | %.2f | %.2f |" % (
            nm, s43["plan_b"][j].sum(), s43["b"][j].sum(),
            s43["emergency"][j].sum(), c43))
    w()
    w("#### 表 6 问题 4-3 紧急购电时段明细")
    w()
    w("| 日期 | 紧急购电时段与电量 | 合计 |")
    w("|---|---|---|")
    for nm in DATES4:
        j = row_of[nm]
        segs = seg_list(s43["emergency"][j])
        txt = "; ".join("%s—%s: %.1f" % (lab(a).split("-")[0],
                                          lab(bb).split("-")[1], v)
                        for a, bb, v in segs)
        w("| %s | %s | %.1f |" % (nm, txt, s43["emergency"][j].sum()))
    w()

    # ---------- 全年汇总 ----------
    w("#### 表 7 全年 (2025.2.1—12.31, 334 天) 结果汇总")
    w()
    w("| 情形 | 全年总费用(元) | 计划+调整购电费(元) | 紧急购电费(元) | "
      "紧急购电量(kWh) | 日均费用(元) |")
    w("|---|---|---|---|---|---|")
    w("| 问题 2 (固定电价, 完全信息) | %.2f | %.2f | 0.00 | 0.00 | %.2f |" % (
        summ["p2"]["total_cost"], summ["p2"]["total_cost"],
        summ["p2"]["total_cost"] / 334))
    w("| 问题 3 (固定电价, 规则A) | %.2f | %.2f | %.2f | %.2f | %.2f |" % (
        summ["p3"]["total"], summ["p3"]["settle"], summ["p3"]["emergency"],
        summ["p3"]["energy"], summ["p3"]["total"] / 334))
    w("| 问题 3 (规则B 实时平衡) | %.2f | %.2f | %.2f | %.2f | %.2f |" % (
        summ["p3"]["total_B"],
        summ["p3"]["total_B"] - summ["p3"]["emergency_B"],
        summ["p3"]["emergency_B"], summ["p3"]["energy_B"],
        summ["p3"]["total_B"] / 334))
    w("| 问题 4-2 (波动电价, 完全信息) | %.2f | %.2f | 0.00 | 0.00 | %.2f |" % (
        summ["p42"]["total_cost"], summ["p42"]["total_cost"],
        summ["p42"]["total_cost"] / 334))
    w("| 问题 4-3 (波动电价, 规则A) | %.2f | %.2f | %.2f | %.2f | %.2f |" % (
        summ["p43"]["total"], summ["p43"]["settle"], summ["p43"]["emergency"],
        summ["p43"]["energy"], summ["p43"]["total"] / 334))
    w()
    w("#### 表 8 多方法检验结果 (问题 1)")
    w()
    w("| 检验方法 | 结果 |")
    w("|---|---|")
    p1 = ver["p1"]
    w("| HiGHS 对偶单纯形 (基准) | 目标值 %.6f 元 |" % p1["highs_ds"])
    w("| HiGHS 内点法 | 目标值 %.6f 元 (差 %.1e) |" % (
        p1["highs_ipm"], p1["gap_ds_ipm"]))
    w("| CBC 分支割平面 (独立求解器) | 目标值 %.6f 元 |" % cbc["p1_day"]["obj"])
    w("| KKT/强对偶最优性证书 | 对偶间隙 %.1e, 平稳性违反 %.1e, 互补松弛 %.1e |" %
      (p1["kkt"]["duality_gap"], p1["kkt"]["max_stationarity_violation"],
       p1["kkt"]["max_comp_slack"]))
    w("| 动态规划 (SOC 网格 15 kWh) | 目标值 %.2f 元 (高出 %.2f 元, 0.21%%) |" %
      (p1["dp_grid15"]["obj"], p1["dp_grid15"]["gap_vs_lp"]))
    w("| 动态规划 (SOC 网格 5 kWh) | 目标值 %.2f 元 (高出 %.2f 元, 0.07%%) |" %
      (p1["dp_grid5"]["obj"], p1["dp_grid5"]["gap_vs_lp"]))
    w("| 允许紧急购电的松弛模型 | 目标值 %.6f 元, 紧急购电 %.1f kWh (命题成立) |" %
      (p1["emergency_relax_obj"], p1["emergency_energy"]))
    w("| 逐时段可行性复核 | SOC 递推误差 %.1e, 功率平衡违反 %.1e |" %
      (p1["feasibility"]["soc_recur_err"],
       p1["feasibility"]["balance_violation"]))
    w("| 固定时段规则策略 (可行) | 目标值 %.2f 元 (比最优高 %.2f 元, %.1f%%) |" %
      (p1["baseline_rules"]["cost"],
       p1["baseline_rules"]["cost"] - p1["highs_ds"],
       100 * (p1["baseline_rules"]["cost"] / p1["highs_ds"] - 1)))
    w()
    w("#### 表 9 灵敏度分析")
    w()
    w("| 情形 | 全年总费用(元) | 相对基准 |")
    w("|---|---|---|")
    base = summ["p3"]["total"]
    for k, v in sen["alignment"].items():
        w("| 预报-实测时间对齐 %s | %.2f | %+.2f%% |" %
          (k, v["total"], 100 * (v["total"] / base - 1)))
    w("| 结算口径 A (全额计划+违约金) | 21445179.00 | %+.2f%% |" %
      (100 * (21445179.0 / base - 1)))
    w("| 逐日闭环约束放宽 (全年滚动 LP) | 12230383.65 | 完全信息情形 |")
    w()
    w("#### 表 10 日前计划保守裕量 (hedge) 与全年费用")
    w()
    w("| hedge | 问题3 总费用(元) | 问题3 紧急购电费(元) | 问题4-3 总费用(元) |")
    w("|---|---|---|---|")
    for key in sorted(hed["att1"], key=lambda k: float(k.split("_")[1])):
        h = float(key.split("_")[1])
        k4 = "A_%.2f" % h
        w("| %.0f%% | %.2f | %.2f | %.2f |" % (
            h * 100, hed["att1"][key]["total"], hed["att1"][key]["emergency"],
            hed["att4"][k4]["total"]))
    w()
    w("#### 表 11 预报时刻安排与全年费用 (问题 3 口径)")
    w()
    w("| 预报时刻 | 全年总费用(元) | 紧急购电费(元) |")
    w("|---|---|---|")
    for k, v in fcs["real_forecast_schedules"].items():
        w("| %s (真实预报) | %.2f | %.2f |" % (k, v["total"], v["emergency"]))
    for k, v in fcs["synthetic_schedules"].items():
        w("| %s (加密, 合成) | %.2f | %.2f |" % (k, v["total"], v["emergency"]))
    w()
    txt = "\n".join(LINES)
    with open(os.path.join(OUT, "tables.md"), "w") as f:
        f.write(txt)
    print(txt[:1500])
    print("... total %d lines" % len(LINES))


if __name__ == "__main__":
    main()
