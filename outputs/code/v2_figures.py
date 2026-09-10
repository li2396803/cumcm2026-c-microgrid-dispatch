"""v2 图表: 策略对比 / 风险-成本前沿 / 寿命成本 / 预报价值 / MPC 典型日."""
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

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "out")
FIG = os.path.join(os.path.dirname(ROOT), "outputs", "figures")

plt.rcParams["font.sans-serif"] = ["Songti SC", "Arial Unicode MS", "STHeiti"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 11
plt.rcParams["figure.dpi"] = 150


def load(f):
    return np.load(os.path.join(OUT, f), allow_pickle=True)


def fig_compare():
    v1 = load("sol_p3.npz")
    saa = load("v2_tree_saa.npz")
    cvar = load("v2_tree_cvar.npz")
    rob = load("v2_tree_robust.npz")
    mpc = load("v2_mpc_b.npz")
    p2 = load("sol_p2.npz")
    names = ["v1\n(规则A)", "v2-A\nSAA树策略", "v2-B\n均值-CVaR", "v2-C\n鲁棒",
             "v2-D\nMPC闭环", "完全信息\n下界"]
    vals = [v1["cost"].sum(), saa["cost"].sum(), cvar["cost"].sum(),
            rob["cost"].sum(), mpc["cost"].sum(), p2["cost"].sum()]
    em = [v1["emergency"].sum(), saa["emergency"].sum(),
          cvar["emergency"].sum(), rob["emergency"].sum(),
          mpc["emergency"].sum(), 0.0]
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6))
    cols = ["#7f8c8d", "#2980b9", "#8e44ad", "#16a085", "#c0392b", "#2c3e50"]
    ax[0].bar(range(6), np.array(vals) / 1e6, color=cols)
    for i, v in enumerate(vals):
        ax[0].text(i, v / 1e6 + .08, "%.2f" % (v / 1e6), ha="center", fontsize=9)
    ax[0].set_xticks(range(6))
    ax[0].set_xticklabels(names, fontsize=9)
    ax[0].set_ylabel("全年总费用 (百万元)")
    ax[0].set_title("各策略全年总费用 (2025.2.1—12.31)")
    ax[0].grid(alpha=.3, axis="y")
    ax[1].bar(range(6), np.array(em) / 1e3, color=cols)
    for i, v in enumerate(em):
        ax[1].text(i, v / 1e3 + 8, "%.0f" % (v / 1e3), ha="center", fontsize=9)
    ax[1].set_xticks(range(6))
    ax[1].set_xticklabels(names, fontsize=9)
    ax[1].set_ylabel("紧急购电量 (千 kWh)")
    ax[1].set_title("紧急购电量对比")
    ax[1].grid(alpha=.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "v2_compare.png"))
    plt.close(fig)


def fig_risk():
    v1 = load("sol_p3.npz")
    saa = load("v2_tree_saa.npz")
    cvar = load("v2_tree_cvar.npz")
    rob = load("v2_tree_robust.npz")
    mpc = load("v2_mpc_b.npz")
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    pts = [("v1 (规则A)", v1, "#7f8c8d", "o"), ("v2-A SAA", saa, "#2980b9", "s"),
           ("v2-B 均值-CVaR", cvar, "#8e44ad", "^"),
           ("v2-C 鲁棒", rob, "#16a085", "v"),
           ("v2-D MPC", mpc, "#c0392b", "D")]
    for lab, d, col, mk in pts:
        c = np.asarray(d["cost"], float)
        x = c.mean() / 1e3
        y = np.sort(c)[int(0.9 * len(c)):].mean() / 1e3
        ax.scatter(x, y, color=col, marker=mk, s=90, label=lab)
        ax.annotate(lab.split()[0], (x, y), textcoords="offset points",
                    xytext=(6, 6), fontsize=9)
    ax.set_xlabel("日均费用 (千元/日)")
    ax.set_ylabel("CVaR90 费用 (千元/日)")
    ax.set_title("风险-成本有效前沿 (样本外 334 天)")
    ax.grid(alpha=.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "v2_risk.png"))
    plt.close(fig)


def fig_degrad():
    with open(os.path.join(OUT, "v2_degrad.json")) as f:
        d = json.load(f)
    x = [r["deg"] for r in d["p2"]]
    e = [r["energy_cost"] / 1e6 for r in d["p2"]]
    w = [r["deg_cost"] / 1e6 for r in d["p2"]]
    t = [(r["energy_cost"] + r["deg_cost"]) / 1e6 for r in d["p2"]]
    life = [100 * r["life_fraction"] for r in d["p2"]]
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.3))
    ax[0].plot(x, e, "o-", color="#2980b9", label="能量费")
    ax[0].plot(x, w, "s-", color="#e67e22", label="寿命费")
    ax[0].plot(x, t, "D-", color="#c0392b", label="合计")
    ax[0].set_xlabel("磨损价 (元/kWh 吞吐)")
    ax[0].set_ylabel("百万元/年")
    ax[0].set_title("寿命成本与全年费用 (问题 2)")
    ax[0].legend()
    ax[0].grid(alpha=.3)
    ax[1].plot(x, life, "o-", color="#16a085")
    ax[1].set_xlabel("磨损价 (元/kWh 吞吐)")
    ax[1].set_ylabel("年寿命消耗 (%/年)")
    ax[1].set_title("等效寿命消耗 (EFC 口径)")
    ax[1].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "v2_degrad.png"))
    plt.close(fig)


def fig_adaptive():
    with open(os.path.join(OUT, "v2_adaptive.json")) as f:
        a = json.load(f)
    obj = np.array(a["obj"])
    subs = a["subsets"]
    base = obj[:, 0]
    gain = (base[:, None] - obj).mean(axis=0)
    names = []
    for s in subs:
        used = [["6:00", "12:00", "18:00"][k] for k in range(3) if s[k]]
        names.append("+".join(used) if used else "不使用更新")
    order = np.argsort(-gain)
    fig, ax = plt.subplots(figsize=(9.6, 4.4))
    ax.bar(range(8), gain[order], color=["#2980b9" if gain[i] > 0 else "#c0392b"
                                         for i in order])
    ax.set_xticks(range(8))
    ax.set_xticklabels([names[i] for i in order], rotation=20, ha="right",
                       fontsize=9)
    ax.axhline(0, color="k", lw=.8)
    ax.set_ylabel("相对不使用更新的费用变化 (元/日, 正=省钱)")
    ax.set_title("各预报子集的日内平均价值 (场景树内评价)")
    ax.grid(alpha=.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "v2_adaptive.png"))
    plt.close(fig)


def fig_mpc_day(day="2025-03-20"):
    data = P.load_data()
    dates = [str(x) for x in data["dates2"]]
    i = dates.index(day)
    mpc = load("v2_mpc_b.npz")
    v1 = load("sol_p3.npz")
    j = list(mpc["dates"]).index(day)
    j1 = list(v1["dates"]).index(day)
    loadE = data["load"][i] * M.DT
    pv = data["pv"][i] * M.DT
    t = np.arange(144) / 6.0
    fig, ax = plt.subplots(3, 1, figsize=(9.2, 7.6), sharex=True)
    ax[0].plot(t, pv * 6, color="#16a085", label="实际光伏")
    ax[0].plot(t, loadE * 6, color="#2c3e50", lw=1, label="小区负载")
    ax[0].set_ylabel("功率 (kW)")
    ax[0].legend(fontsize=9)
    ax[1].plot(t, v1["b"][j1] * 6, color="#7f8c8d", label="v1 调整后购电")
    ax[1].plot(t, mpc["b"][j] * 6, color="#c0392b", label="v2 MPC 购电")
    ax[1].fill_between(t, 0, mpc["emergency"][j] * 6, color="#e67e22", alpha=.6,
                       label="MPC 紧急购电")
    ax[1].set_ylabel("功率 (kW)")
    ax[1].legend(fontsize=9, ncol=3)
    sv1 = np.concatenate([[6000.0], v1["s"][j1]])
    sm = 6000.0 + np.cumsum(0.9 * mpc["c"][j] - mpc["d"][j] / 0.9)
    sm = np.concatenate([[6000.0], sm])
    ax[2].plot(np.arange(145) / 6, sv1, color="#7f8c8d", label="v1 储电量")
    ax[2].plot(np.arange(145) / 6, sm, color="#c0392b", label="v2 MPC 储电量")
    ax[2].axhline(10800, ls="--", color="gray", lw=.8)
    ax[2].axhline(1200, ls="--", color="gray", lw=.8)
    ax[2].set_ylabel("储电量 (kWh)")
    ax[2].set_xlabel("时刻 (小时)")
    ax[2].legend(fontsize=9)
    ax[0].set_title(f"v2 MPC 闭环策略典型日对比: {day}")
    for a in ax:
        a.grid(alpha=.3)
        a.set_xlim(0, 24)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "v2_mpc_day.png"))
    plt.close(fig)


def main():
    os.makedirs(FIG, exist_ok=True)
    fig_compare()
    fig_risk()
    fig_degrad()
    fig_adaptive()
    fig_mpc_day()
    print("v2 figures ->", FIG)


if __name__ == "__main__":
    main()
