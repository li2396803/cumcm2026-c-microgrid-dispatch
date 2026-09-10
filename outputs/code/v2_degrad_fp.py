"""v2 改进 3(续): 线性寿命成本系数的自洽标定 (不动点迭代).

线性代理 c_deg 与雨流寿命模型并不等价 (后者为 Wöhler 指数 k 的幂律)。
以"实际平均寿命成本/吞吐"作为下一轮 c_deg, 迭代至自洽:
    c^{(t+1)} = (雨流寿命成本 at c^{(t)}) / (SOC 吞吐 at c^{(t)})
并对比 k=1(等价线性) 与 k=1.6(浅循环更省) 两种寿命模型下的最优策略。
"""
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


def solve_year(deg, p, load, pv, idx):
    B, C, D, S = [], [], [], []
    for i in idx:
        sol = M.solve_day(p, load[i] * M.DT, pv[i] * M.DT, deg_cost=deg,
                          solver="highs-ds")
        B.append(sol["b"]); C.append(sol["c"]); D.append(sol["d"])
        S.append(sol["s"])
    return np.array(B), np.array(C), np.array(D), np.array(S)


def rainflow_year(S, k=1.6):
    # 全年轨迹拼接后一次性雨流计数 (每日闭环, 残余半循环可跨日配对)
    traj = np.concatenate([[6000.0]] + [np.asarray(s) for s in S])
    cyc, rem = LT.rainflow(traj)
    cap = LT.COST_PER_KWH * LT.CAP_INSTALLED
    dod = np.clip(cyc / LT.CAP_USABLE, 1e-6, 1.0)
    dodr = np.clip(rem / LT.CAP_USABLE, 1e-6, 1.0)
    life = float(((dod / LT.DOD_REF) ** k / LT.N0).sum()
                 + 0.5 * ((dodr / LT.DOD_REF) ** k / LT.N0).sum())
    deg = float((cap / (LT.N0 * (dod / LT.DOD_REF) ** (-k))).sum()
                + 0.5 * (cap / (LT.N0 * (dodr / LT.DOD_REF) ** (-k))).sum())
    allc = cyc
    return {"life_fraction": float(life), "deg_cost": float(deg),
            "n_cycles": int(len(allc)),
            "depth_p50": float(np.median(allc)) if len(allc) else 0.0,
            "depth_p90": float(np.percentile(allc, 90)) if len(allc) else 0.0,
            "depth_max": float(allc.max()) if len(allc) else 0.0}


def main():
    data = P.load_data()
    idx = P._day_range(data)
    p = data["att1"][:, 0]
    load, pv = data["load"], data["pv"]
    out = {"iter": []}
    deg = 0.0
    for it in range(6):
        B, C, D, S = solve_year(deg, p, load, pv, idx)
        thr = float(np.sum(M.ETA * C + D / M.ETA))
        energy = float(np.sum(p[None, :] * B))
        rf16 = rainflow_year(S, 1.6)
        rf10 = rainflow_year(S, 1.0)
        deg_next = rf16["deg_cost"] / thr if thr > 0 else 0.0
        out["iter"].append({"it": it, "c_deg": deg, "energy_cost": energy,
                            "throughput": thr, "rf_k16": rf16, "rf_k10": rf10,
                            "c_deg_next": deg_next,
                            "total_true": energy + rf16["deg_cost"],
                            "total_linear": energy + deg * thr})
        print(f"iter {it}: c_deg={deg:.4f} 能量费={energy:,.0f} 吞吐={thr:,.0f} "
              f"雨流成本(k=1.6)={rf16['deg_cost']:,.0f} 寿命={100*rf16['life_fraction']:.2f}%/年 "
              f"自洽c={deg_next:.4f} 真实合计={energy+rf16['deg_cost']:,.0f}",
              flush=True)
        if abs(deg_next - deg) < 0.002:
            break
        deg = deg_next
    with open(os.path.join(OUT, "v2_degrad_fp.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("saved v2_degrad_fp.json")


if __name__ == "__main__":
    main()
