"""v2 改进 3 计算: 储能寿命成本对最优策略与全年费用的影响 (P1/P2/P3)."""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model as M
import pipeline as P
import v2_lifetime as LT

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "out")


def annual_stats(S, C, D, dates):
    """等效满循环 (EFC) 寿命核算.

    EFC = SOC 吞吐 / (2 * 可用容量);  寿命按 N0=6000 次满循环 (90% DoD) 计,
    年寿命消耗 = EFC / N0,  年寿命成本 = 资本成本 * EFC / N0。
    该口径与线性磨损成本模型 (k=1) 完全自洽。
    """
    thr = float(np.sum(M.ETA * C + D / M.ETA))
    efc = thr / (2.0 * LT.CAP_USABLE)
    cap = LT.COST_PER_KWH * LT.CAP_INSTALLED
    life = efc / LT.N0
    return {"throughput": thr, "efc": float(efc),
            "life_fraction": float(life),
            "deg_cost": float(cap * life),
            "c_deg_theory": float(cap / (LT.N0 * 2.0 * LT.CAP_USABLE))}


def run_p2(deg):
    data = P.load_data()
    idx = P._day_range(data)
    p = data["att1"][:, 0]
    B, C, D, S = [], [], [], []
    cost = 0.0
    for i in idx:
        sol = M.solve_day(p, data["load"][i] * M.DT, data["pv"][i] * M.DT,
                          s_init=6000.0, s_end=6000.0, deg_cost=deg,
                          solver="highs-ds")
        B.append(sol["b"]); C.append(sol["c"]); D.append(sol["d"])
        S.append(sol["s"]); cost += sol["obj"]
    B, C, D, S = map(np.array, (B, C, D, S))
    st = annual_stats(S, C, D, None)
    return {"deg": deg, "energy_cost": float(np.sum(p[None, :] * B)),
            "total_obj": float(cost), **st}


def run_p1(deg):
    data = P.load_data()
    a1 = data["att1"]
    p, loadE, pvE = a1[:, 0], a1[:, 1] * M.DT, a1[:, 2] * M.DT
    sol = M.solve_day(p, loadE, pvE, deg_cost=deg, solver="highs-ds")
    traj = np.concatenate([[6000.0], sol["s"]])
    c, d = sol["c"], sol["d"]
    thr = float(np.sum(M.ETA * c + d / M.ETA))
    efc = thr / (2.0 * LT.CAP_USABLE)
    life = efc / LT.N0
    cap = LT.COST_PER_KWH * LT.CAP_INSTALLED
    return {"deg": deg, "energy_cost": float(np.sum(p * sol["b"])),
            "charge": float(sol["c"].sum()), "discharge": float(sol["d"].sum()),
            "throughput": thr, "efc": float(efc),
            "life_fraction": float(life), "deg_cost": float(cap * life),
            "c_deg_theory": float(cap / (LT.N0 * 2.0 * LT.CAP_USABLE))}


def main():
    grid = [0.0, 0.05, 0.10, 0.20, 0.30]
    out = {"p1": [], "p2": []}
    for g in grid:
        r1 = run_p1(g)
        out["p1"].append(r1)
        print("P1 deg=%.2f 能量费 %8.0f 吞吐 %8.0f kWh 真实寿命费 %7.0f 元"
              % (g, r1["energy_cost"], r1["throughput"], r1["deg_cost_true"]),
              flush=True)
    for g in grid:
        r2 = run_p2(g)
        r2["total_with_true_deg"] = r2["energy_cost"] + r2["deg_cost"]
        out["p2"].append(r2)
        print("P2 deg=%.2f 能量费 %10.0f 吞吐 %9.0f kWh 真实寿命费 %8.0f 元 "
              "合计 %10.0f 元 (寿命消耗 %.2f%%/年) -> 等效年成本 %10.0f"
              % (g, r2["energy_cost"], r2["throughput"], r2["deg_cost"],
                 r2["total_with_true_deg"], 100 * r2["life_fraction"],
                 r2["deg_cost"] / max(r2["life_fraction"], 1e-9) *
                 min(r2["life_fraction"], 1.0)), flush=True)
    with open(os.path.join(OUT, "v2_degrad.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("saved v2_degrad.json")


if __name__ == "__main__":
    main()
