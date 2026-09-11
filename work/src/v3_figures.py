"""v3 图表：修正后的问题 2／4-2 费用曲线、预报信息价值对比、策略总览。"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model as M
import pipeline as P

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "out")
FIG = os.path.join(os.path.dirname(ROOT), "outputs", "figures")

plt.rcParams["font.sans-serif"] = ["Songti SC", "Arial Unicode MS", "STHeiti"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 11
plt.rcParams["figure.dpi"] = 150


def L(f):
    return np.load(os.path.join(OUT, f), allow_pickle=True)


def fig3_v3():
    """问题 2 / 4-2 逐日与逐月费用（v3 修正后）。"""
    p2, p42 = L("p2_v3.npz"), L("p42_v3.npz")
    dates = [str(x) for x in p2["dates"]]
    x = np.arange(len(dates))
    fig, ax = plt.subplots(2, 1, figsize=(9.6, 6.6))
    ax[0].plot(x, p2["plan_cost"], lw=.7, color="#2980b9", label="问题 2 计划购电费")
    ax[0].plot(x, p2["em_cost"], lw=.7, color="#c0392b", label="问题 2 紧急购电费")
    ax[0].plot(x, p2["cost"], lw=.9, color="#2c3e50", label="问题 2 合计")
    ax[0].plot(x, p42["cost"], lw=.7, color="#e67e22", label="问题 4-2 合计（实时电价）")
    ax[0].set_ylabel("日费用 (元)")
    ax[0].legend(fontsize=9, ncol=2)
    ax[0].grid(alpha=.3)
    month = np.array([int(d[5:7]) for d in dates])
    ms_plan = [p2["plan_cost"][month == m].sum() for m in range(2, 13)]
    ms_em = [p2["em_cost"][month == m].sum() for m in range(2, 13)]
    ax[1].bar(np.arange(11), np.array(ms_plan) / 1e4, color="#2980b9", label="计划购电费")
    ax[1].bar(np.arange(11), np.array(ms_em) / 1e4, bottom=np.array(ms_plan) / 1e4,
              color="#c0392b", label="紧急购电费")
    ax[1].set_xticks(range(11))
    ax[1].set_xticklabels([f"{m}月" for m in range(2, 13)])
    ax[1].set_ylabel("月费用 (万元)")
    ax[1].legend(fontsize=9)
    ax[1].grid(alpha=.3, axis="y")
    ax[0].set_title("问题 2 / 问题 4-2 费用构成（v3 修正后：计划用预报、结算用实际）")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig3_p2_cost.png"))
    plt.close(fig)


def fig8_info_value():
    """预报信息价值的对比 + 紧急购电的时间分布。"""
    p2 = L("p2_v3.npz")
    dates = [str(x) for x in p2["dates"]]
    month = np.array([int(d[5:7]) for d in dates])
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6))
    names = ["理想情形\n(已知实际光伏)", "问题 2\n(典型日预报)", "0:00 每日预报\n(附件3, 不调整)",
             "问题 3\n(每日预报+调整)"]
    vals = [12_245_047, 18_092_018, 17_398_560, 16_970_919]
    cols = ["#2c3e50", "#c0392b", "#e67e22", "#27ae60"]
    ax[0].bar(range(4), np.array(vals) / 1e6, color=cols)
    for i, v in enumerate(vals):
        ax[0].text(i, v / 1e6 + .12, f"{v/1e6:.2f}", ha="center", fontsize=9)
    ax[0].set_xticks(range(4)); ax[0].set_xticklabels(names, fontsize=8.5)
    ax[0].set_ylabel("全年总费用 (百万元)")
    ax[0].set_title("预报信息质量的价值（334 天）")
    ax[0].grid(alpha=.3, axis="y")
    em_m = [p2["em_energy"][month == m].sum() for m in range(2, 13)]
    ax[1].bar(range(11), np.array(em_m) / 1e3, color="#c0392b")
    ax[1].set_xticks(range(11)); ax[1].set_xticklabels([f"{m}月" for m in range(2, 13)])
    ax[1].set_ylabel("紧急购电量 (千 kWh)")
    ax[1].set_title("问题 2 紧急购电量的月度分布（冬季典型日预报高估光伏）")
    ax[1].grid(alpha=.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "v3_p2_info_value.png"))
    plt.close(fig)


def fig7_v3():
    """各策略全年总费用总览（v3）。"""
    v1p3 = L("sol_p3.npz"); saa = L("v2_tree_saa.npz")
    mpc = L("v2_mpc_b.npz"); p2 = L("p2_v3.npz"); p42 = L("p42_v3.npz")
    p3 = L("sol_p3.npz"); p43 = L("sol_p43.npz"); mpc4 = L("v2_mpc_att4.npz")
    ideal = L("sol_p2.npz")
    names = ["问题2\n(v3 修正)", "问题4-2\n(v3 修正)", "问题3\n(v1 基线)",
             "问题3\n(v2-D MPC)", "问题4-3\n(v1 基线)", "问题4-3\n(v2-D MPC)",
             "理想下界\n(已知实际光伏)"]
    vals = [p2["cost"].sum(), p42["cost"].sum(), p3["cost"].sum(), mpc["cost"].sum(),
            p43["cost"].sum(), mpc4["cost"].sum(), ideal["cost"].sum()]
    cols = ["#c0392b", "#e67e22", "#7f8c8d", "#2980b9", "#7f8c8d", "#16a085", "#2c3e50"]
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.bar(range(len(vals)), np.array(vals) / 1e6, color=cols)
    for i, v in enumerate(vals):
        ax.text(i, v / 1e6 + .12, f"{v/1e6:.2f}", ha="center", fontsize=9)
    ax.set_xticks(range(len(vals))); ax.set_xticklabels(names, fontsize=8.5)
    ax.set_ylabel("全年总费用 (百万元)")
    ax.set_title("四种问题与两类改进的全年费用总览（2025.2.1—12.31，334 天）")
    ax.grid(alpha=.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig7_scenarios.png"))
    plt.close(fig)


def main():
    fig3_v3(); fig8_info_value(); fig7_v3()
    print("v3 figures ->", FIG)


if __name__ == "__main__":
    main()
