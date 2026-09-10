"""生成说明文档所需图表 (中文字体)."""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model as M
import pipeline as P
from execution import run_interval_ruleA, run_interval_ruleB

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "out")
FIG = os.path.join(os.path.dirname(ROOT), "outputs", "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Songti SC", "Arial Unicode MS", "STHeiti"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 11
plt.rcParams["figure.dpi"] = 150


def hours():
    return np.arange(144) / 6.0


def fig1(data):
    a1 = data["att1"]
    t = hours()
    fig, ax = plt.subplots(3, 1, figsize=(9, 7.2), sharex=True)
    ax[0].plot(t, a1[:, 0], color="#c0392b")
    ax[0].set_ylabel("电价 (元/kWh)")
    ax[0].set_title("附件 1 代表日数据: 电价 / 小区负载 / 光伏预测功率")
    ax[1].plot(t, a1[:, 1], color="#2c3e50")
    ax[1].set_ylabel("负载 (kW)")
    ax[2].plot(t, a1[:, 2], color="#e67e22")
    ax[2].set_ylabel("光伏 (kW)")
    ax[2].set_xlabel("时刻 (小时)")
    for a in ax:
        a.grid(alpha=.3)
    for a in ax[:-1]:
        a.set_xlim(0, 24)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig1_data_profile.png"))
    plt.close(fig)


def fig2(data, sol):
    a1 = data["att1"]
    t = hours()
    fig, ax = plt.subplots(3, 1, figsize=(9, 7.6), sharex=True)
    ax[0].plot(t, a1[:, 0], color="#7f8c8d", label="电价")
    ax[0].set_ylabel("电价 (元/kWh)")
    ax[0].legend(loc="upper left")
    ax[1].fill_between(t, 0, sol["b"] * 6, color="#3498db", alpha=.65,
                       label="购电量(折算功率)")
    ax[1].fill_between(t, 0, -sol["c"] * 6, color="#27ae60", alpha=.65,
                       label="充电")
    ax[1].fill_between(t, 0, sol["d"] * 6, color="#e74c3c", alpha=.65,
                       label="放电")
    ax[1].plot(t, a1[:, 1] - a1[:, 2], color="k", lw=1.2,
               label="净负荷 (负载-光伏)")
    ax[1].axhline(0, color="k", lw=.6)
    ax[1].set_ylabel("功率 (kW)")
    ax[1].legend(ncol=2, fontsize=9, loc="upper left")
    ax[2].plot(t, sol["s"], color="#8e44ad")
    ax[2].axhline(10800, ls="--", color="gray", lw=.8)
    ax[2].axhline(1200, ls="--", color="gray", lw=.8)
    ax[2].axhline(6000, ls=":", color="gray", lw=.8)
    ax[2].set_ylabel("储电量 (kWh)")
    ax[2].set_xlabel("时刻 (小时)")
    for a in ax:
        a.grid(alpha=.3)
        a.set_xlim(0, 24)
    ax[1].set_title("问题 1: 最优购电/充放电策略与储电量轨迹 (购电费 35126.95 元)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig2_p1_strategy.png"))
    plt.close(fig)


def fig3(data, sol2, sol42):
    dates = [str(x) for x in sol2["dates"]]
    x = np.arange(len(dates))
    fig, ax = plt.subplots(2, 1, figsize=(9.5, 6.4))
    ax[0].plot(x, sol2["cost"], lw=.8, color="#2980b9", label="固定电价 (问题2)")
    ax[0].plot(x, sol42["cost"], lw=.8, color="#e67e22",
               label="波动电价 (问题4-2)")
    ax[0].set_ylabel("日购电费 (元)")
    ax[0].legend()
    for a in ax:
        a.grid(alpha=.3)
    month = np.array([int(d[5:7]) for d in dates])
    ms = [sol2["cost"][month == m].sum() for m in range(2, 13)]
    ms2 = [sol42["cost"][month == m].sum() for m in range(2, 13)]
    ax[1].bar(np.arange(11) - .2, ms, width=.4, color="#2980b9",
              label="固定电价 (问题2)")
    ax[1].bar(np.arange(11) + .2, ms2, width=.4, color="#e67e22",
              label="波动电价 (问题4-2)")
    ax[1].set_xticks(range(11))
    ax[1].set_xticklabels([f"{m}月" for m in range(2, 13)])
    ax[1].set_ylabel("月购电费 (元)")
    ax[1].legend()
    ax[0].set_title("问题 2 / 问题 4-2: 逐日与逐月购电费用")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig3_p2_cost.png"))
    plt.close(fig)


def fig4(data, day="2025-03-20"):
    dates = [str(x) for x in data["dates2"]]
    i = dates.index(day)
    p = data["att1"][:, 0]
    loadE = data["load"][i] * M.DT
    pv_act = data["pv"][i] * M.DT
    f0 = P.forecast_energy(data, i, 0)
    plan = P.plan_day(p, loadE, f0, 6000, 6000)
    r = P.problem3(data, "att1")
    d = {x["date"]: x for x in r}[day]
    t = hours()
    fig, ax = plt.subplots(3, 1, figsize=(9, 7.6), sharex=True)
    ax[0].plot(t, f0 * 6, color="#e67e22", label="日前预报光伏")
    ax[0].plot(t, pv_act * 6, color="#16a085", label="实际光伏")
    ax[0].plot(t, loadE * 6, color="#2c3e50", lw=1, label="小区负载")
    ax[0].set_ylabel("功率 (kW)")
    ax[0].legend(fontsize=9, ncol=2)
    ax[1].plot(t, d["plan_b"] * 6, color="#2980b9", label="日前计划购电")
    ax[1].plot(t, d["b"] * 6, color="#8e44ad", ls="--", label="调整后购电")
    ax[1].fill_between(t, 0, d["emergency"] * 6, color="#c0392b", alpha=.6,
                       label="紧急购电")
    ax[1].set_ylabel("功率 (kW)")
    ax[1].legend(fontsize=9, ncol=3)
    ax[2].plot(t, d["s"], color="#8e44ad")
    ax[2].axhline(10800, ls="--", color="gray", lw=.8)
    ax[2].axhline(1200, ls="--", color="gray", lw=.8)
    ax[2].set_ylabel("储电量 (kWh)")
    ax[2].set_xlabel("时刻 (小时)")
    ax[0].set_title(f"问题 3: {day} 计划/调整/紧急购电与储能轨迹")
    for a in ax:
        a.grid(alpha=.3)
        a.set_xlim(0, 24)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig4_p3_day.png"))
    plt.close(fig)


def fig5():
    with open(os.path.join(OUT, "analysis_hedge.json")) as f:
        hd = json.load(f)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for mode, lab, col in [("att1", "固定电价 (问题3)", "#2980b9"),
                           ("att4", "波动电价 (问题4-3)", "#e67e22")]:
        xs, ys = [], []
        for key, val in hd[mode].items():
            xs.append(float(key.split("_")[1]))
            ys.append(val["total"] / 1e6)
        o = np.argsort(xs)
        ax.plot(np.array(xs)[o] * 100, np.array(ys)[o], "o-", color=col,
                label=lab)
    ax.set_xlabel("日前计划的光伏保守裕量 (%)")
    ax.set_ylabel("全年总费用 (百万元)")
    ax.set_title("日前计划保守程度与全年总费用的关系")
    ax.grid(alpha=.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig5_hedge.png"))
    plt.close(fig)


def fig6():
    with open(os.path.join(OUT, "analysis_forecast.json")) as f:
        fd = json.load(f)
    names, vals = [], []
    for k, v in fd["real_forecast_schedules"].items():
        names.append(k)
        vals.append(v["total"] / 1e6)
    for k, v in fd["synthetic_schedules"].items():
        if "真实" in k:
            continue
        names.append(k)
        vals.append(v["total"] / 1e6)
    fig, ax = plt.subplots(figsize=(9.6, 4.6))
    cols = ["#2980b9"] * len(fd["real_forecast_schedules"]) + \
           ["#e67e22"] * (len(names) - len(fd["real_forecast_schedules"]))
    ax.bar(range(len(names)), vals, color=cols)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=18, ha="right", fontsize=9)
    for i, v in enumerate(vals):
        ax.text(i, v + .05, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_ylabel("全年总费用 (百万元)")
    ax.set_title("预报时刻安排对全年总费用的影响 (蓝: 真实预报; 橙: 加密预报)")
    ax.grid(alpha=.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig6_forecast_value.png"))
    plt.close(fig)


def fig7():
    with open(os.path.join(OUT, "summary.json")) as f:
        s = json.load(f)
    labels = ["问题2\n(完全信息)", "问题3\n(规则A)", "问题3\n(规则B)",
              "问题3\n(保守15%)", "问题4-2\n(波动电价)", "问题4-3\n(规则A)",
              "问题4-3\n(规则B)"]
    with open(os.path.join(OUT, "analysis_hedge.json")) as f:
        hd = json.load(f)
    vals = [s["p2"]["total_cost"], s["p3"]["total"], s["p3"]["total_B"],
            hd["att1"]["A_0.15"]["total"], s["p42"]["total_cost"],
            s["p43"]["total"], s["p43"]["total_B"]]
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    cols = ["#2c3e50", "#c0392b", "#e67e22", "#27ae60", "#2c3e50",
            "#c0392b", "#e67e22"]
    ax.bar(range(len(vals)), np.array(vals) / 1e6, color=cols)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, fontsize=9)
    for i, v in enumerate(vals):
        ax.text(i, v / 1e6 + .1, f"{v/1e6:.2f}", ha="center", fontsize=9)
    ax.set_ylabel("全年总费用 (百万元)")
    ax.set_title("各模型/策略下的全年购电总费用对比 (2025.2.1—12.31)")
    ax.grid(alpha=.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig7_scenarios.png"))
    plt.close(fig)


def main():
    data = P.load_data()
    s1 = np.load(os.path.join(OUT, "sol_p1.npz"))
    s2 = np.load(os.path.join(OUT, "sol_p2.npz"), allow_pickle=True)
    s42 = np.load(os.path.join(OUT, "sol_p42.npz"), allow_pickle=True)
    fig1(data)
    fig2(data, s1)
    fig3(data, s2, s42)
    fig4(data)
    fig5()
    fig6()
    fig7()
    print("figures ->", FIG)
    print(sorted(os.listdir(FIG)))


if __name__ == "__main__":
    main()
