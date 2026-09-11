"""v3: 修正问题 2 / 4-2 的信息结构（计划用预报，缺口用实际结算）。

修正要点（v1/v2 的错误）:
  v1/v2 把附件2 的"光伏发电实际功率"直接当成日前计划的输入, 于是计划与执行同源,
  紧急购电恒为 0, 使题面要求的表 3（紧急购电）失去意义。

修正后的信息结构:
  * 电价          : 附件1（每天相同）
  * 小区负载      : 附件2 的实际负载 —— 视为已知（题面只对光伏给预报, P3 同样只预报光伏）
  * 日前光伏预报  : 附件1 的"光伏发电预测功率"列（典型日/年均剖面, 每天相同）
  * 光伏实际出力  : 附件2 的"光伏发电实际功率"
  => 计划购电量按预报制定, 实际不足部分按 5 倍价紧急购电。
"""
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
DATES4 = ["2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21"]


def run(price_mode="att1", s0=6000.0, s_end=6000.0):
    data = P.load_data()
    idx = P._day_range(data)
    load, pv = data["load"], data["pv"]
    p_fix = data["att1"][:, 0]
    pv_fc = data["att1"][:, 2] * M.DT          # 附件1 的预测功率 → 日前预报 (kWh/时段)
    out = {"dates": [], "plan_cost": [], "em_cost": [], "em_energy": [],
           "total": [], "b": [], "c": [], "d": [], "s": [], "emergency": [],
           "spill": [], "plan_b": []}
    for i in idx:
        p = p_fix if price_mode == "att1" else data["price_rt"][i]
        loadE = load[i] * M.DT
        pv_act = pv[i] * M.DT
        sol = M.solve_day(p, loadE, pv_fc, s_init=s0, s_end=s_end,
                          solver="highs-ds")
        sim = M.simulate_execution(sol["b"], sol["c"], sol["d"], p, loadE, pv_act)
        out["dates"].append(str(data["dates2"][i]))
        out["plan_cost"].append(float(np.sum(p * sol["b"])))
        out["em_cost"].append(sim["emergency_cost"])
        out["em_energy"].append(sim["emergency_energy"])
        out["total"].append(float(np.sum(p * sol["b"])) + sim["emergency_cost"])
        out["plan_b"].append(sol["b"])
        out["b"].append(sol["b"])          # 计划即执行（P2 无调整）
        out["c"].append(sol["c"])
        out["d"].append(sol["d"])
        out["s"].append(sol["s"])
        out["emergency"].append(sim["e"])
        out["spill"].append(sim["spill"])
    return out


def summarize(r, tag):
    tot = float(np.sum(r["total"]))
    plan = float(np.sum(r["plan_cost"]))
    em = float(np.sum(r["em_cost"]))
    emk = float(np.sum(r["em_energy"]))
    days = len(r["dates"])
    print(f"[{tag}] {days} 天: 计划购电费 {plan:,.0f} 元 + 紧急购电费 {em:,.0f} 元 "
          f"= {tot:,.0f} 元; 紧急购电量 {emk:,.0f} kWh "
          f"({100*emk/ (sum(x.sum() for x in r['b'])):.2f}% of 购电量)")
    return dict(days=days, plan_cost=plan, em_cost=em, total=tot, em_energy=emk)


def main():
    res = {}
    for mode, tag in [("att1", "问题2 (固定电价, 附件1电价)"),
                      ("att4", "问题4-2 (波动电价, 附件4电价)")]:
        r = run(mode)
        key = "p2_v3" if mode == "att1" else "p42_v3"
        np.savez_compressed(
            os.path.join(OUT, key + ".npz"),
            dates=np.array(r["dates"]), plan_b=np.array(r["plan_b"]),
            b=np.array(r["b"]), c=np.array(r["c"]), d=np.array(r["d"]),
            s=np.array(r["s"]), emergency=np.array(r["emergency"]),
            cost=np.array(r["total"]), plan_cost=np.array(r["plan_cost"]),
            em_cost=np.array(r["em_cost"]), em_energy=np.array(r["em_energy"]))
        res[mode] = summarize(r, tag)
        res[mode]["by_date"] = {
            d: {"em_energy": float(r["em_energy"][j]),
                "em_cost": float(r["em_cost"][j]),
                "total": float(r["total"][j])}
            for j, d in enumerate(r["dates"]) if d in DATES4}
        res[mode]["n_days_with_em"] = int(np.sum(np.array(r["em_energy"]) > 1e-6))
        res[mode]["em_share"] = float(np.sum(r["em_energy"]) /
                                      np.sum([x.sum() for x in r["b"]]))
    with open(os.path.join(OUT, "v3_p2_summary.json"), "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1)[:1200])


if __name__ == "__main__":
    main()
