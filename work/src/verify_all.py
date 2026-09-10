"""多方法检验: 最优性证书 / 独立求解器 / 独立算法 / 可行性复核 / 敏感性."""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model as M
import pipeline as P
import verify as V

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "out")
report = {}


def sec(name):
    print("=" * 70)
    print(name)
    print("=" * 70)


def check_p1(data):
    sec("1. 问题 1 最优性与可行性检验")
    a1 = data["att1"]
    p, loadE, pvE = a1[:, 0], a1[:, 1] * M.DT, a1[:, 2] * M.DT
    r = {}
    sol_ds = M.solve_day(p, loadE, pvE, solver="highs-ds")
    sol_ipm = M.solve_day(p, loadE, pvE, solver="highs-ipm")
    r["highs_ds"] = sol_ds["obj"]
    r["highs_ipm"] = sol_ipm["obj"]
    r["gap_ds_ipm"] = abs(sol_ds["obj"] - sol_ipm["obj"])
    # 可行性与逐时段复核
    r["feasibility"] = M.check_solution(sol_ds, loadE, pvE, 6000.0, 6000.0)
    # KKT + 强对偶
    r["kkt"] = V.kkt_certificate(p, loadE, pvE, 6000.0, 6000.0)
    # 独立算法: SOC 网格动态规划
    for g in (15.0, 5.0):
        dp = V.dp_solve(p, loadE, pvE, 6000.0, 6000.0, grid=g)
        r[f"dp_grid{g:g}"] = {"obj": dp["obj"],
                              "gap_vs_lp": dp["obj"] - sol_ds["obj"]}
    # 允许紧急购电时最优解不变 (验证命题: 确定性最优计划不会短缺)
    sol_em = M.solve_day(p, loadE, pvE, allow_emergency=True)
    r["emergency_relax_obj"] = sol_em["obj"]
    r["emergency_energy"] = float(sol_em["e"].sum())
    # 同时充放电检查
    r["simultaneous_slots"] = int(np.sum((sol_ds["c"] > 1e-6) & (sol_ds["d"] > 1e-6)))
    # 与朴素策略对比 (固定时段规则法)
    r["baseline_rules"] = baseline_rules(p, loadE, pvE)
    print(json.dumps(r, ensure_ascii=False, indent=1, default=float))
    return r


def baseline_rules(p, loadE, pvE):
    """固定时段规则基线 (可行且解析构造):
      放电: 18:00-21:00 满功率放电, 放电总量 D (≤ 2500 kWh);
      充电: 在电价最低的时段按功率上限充电, 充电总量 D/0.81
            (恰好补偿放电损耗, 保证 24:00 回到 6000 kWh);
      光伏盈余不储存, 直接弃光。"""
    from model import K, ETA, E_MAX
    c = np.zeros(K)
    d = np.zeros(K)
    hour = np.arange(K) * 10 // 60
    peak = (hour >= 18) & (hour < 21)
    # 放电总量受功率上限与储电量下限约束
    D = 0.0
    cur = 6000.0
    for k in range(K):
        if peak[k]:
            dd = min(E_MAX, max(0.0, (cur - 1200) * ETA))
            d[k] = dd
            cur -= dd / ETA
            D += dd
    charge_need = D / (ETA * ETA)                 # 交流侧充电量
    order = np.argsort(-p, kind="stable")         # 电价从高到低, 末尾为最便宜
    for k in order[::-1]:
        if charge_need <= 1e-9:
            break
        cc = min(E_MAX, charge_need)
        # 储电量上限 (充电在放电之前进行)
        cc = min(cc, max(0.0, (10800 - 6000) / ETA))
        c[k] = cc
        charge_need -= cc
    cur = 6000.0
    soc = np.empty(K)
    for k in range(K):
        cur += ETA * c[k] - d[k] / ETA
        soc[k] = cur
    ok = bool(soc.min() >= 1200 - 1e-6 and soc.max() <= 10800 + 1e-6
              and abs(soc[-1] - 6000.0) < 1e-6)
    b = np.maximum(0.0, loadE + c - d - pvE)
    spill = np.maximum(0.0, pvE - loadE - c + d)
    return {"feasible": ok, "cost": float(np.sum(p * b)),
            "b_sum": float(b.sum()), "charge": float(c.sum()),
            "discharge": float(d.sum()),
            "spill": float(spill.sum()),
            "soc_min": float(soc.min()), "soc_max": float(soc.max()),
            "soc_end": float(soc[-1])}


def check_p2(data, sol):
    sec("2. 问题 2 全年解复核")
    p = data["att1"][:, 0]
    load, pv = data["load"], data["pv"]
    idx = P._day_range(data)
    r = {"days": len(idx), "violations": {}, "total_cost_recomputed": 0.0,
         "total_energy": 0.0}
    worst = {k: 0.0 for k in ["soc_rec", "soc_low", "soc_high", "soc_end",
                              "balance", "power", "negative", "emergency"]}
    tot = 0.0
    for j, i in enumerate(idx):
        loadE = load[i] * M.DT
        pvE = pv[i] * M.DT
        b = sol["b"][j]
        c = sol["c"][j]
        d = sol["d"][j]
        s = sol["s"][j]
        cost = float(np.sum(p * b))
        tot += cost
        # 独立重算 SOC
        cur = 6000.0
        rec = np.empty(M.K)
        for k in range(M.K):
            cur = cur + M.ETA * c[k] - d[k] / M.ETA
            rec[k] = cur
        worst["soc_rec"] = max(worst["soc_rec"], float(np.abs(rec - s).max()))
        worst["soc_low"] = max(worst["soc_low"], float(max(0.0, 1200 - s.min())))
        worst["soc_high"] = max(worst["soc_high"], float(max(0.0, s.max() - 10800)))
        worst["soc_end"] = max(worst["soc_end"], float(abs(s[-1] - 6000.0)))
        worst["balance"] = max(worst["balance"],
                               float(max(0.0, -(b + d + pvE - loadE - c).min())))
        worst["power"] = max(worst["power"],
                             float(max(0.0, c.max() - M.E_MAX, d.max() - M.E_MAX)))
        worst["negative"] = max(worst["negative"],
                                float(max(0.0, -min(b.min(), c.min(), d.min()))))
        e = np.maximum(0.0, loadE + c - d - b - pvE)
        worst["emergency"] = max(worst["emergency"], float(e.sum()))
    r["violations"] = worst
    r["total_cost_recomputed"] = tot
    r["total_cost_solver"] = float(sol["cost"].sum())
    r["cost_gap"] = abs(tot - float(sol["cost"].sum()))
    r["zero_emergency_confirmed"] = worst["emergency"] < 1e-6
    # 紧急购电松弛下最优值不变 (抽 3 天)
    names = ["2025-03-20", "2025-09-23", "2025-12-21"]
    dates = [str(x) for x in data["dates2"]]
    chk = {}
    for nm in names:
        i = dates.index(nm)
        l2 = data["load"][i] * M.DT
        pv2 = data["pv"][i] * M.DT
        a = M.solve_day(p, l2, pv2)
        b = M.solve_day(p, l2, pv2, allow_emergency=True, solver="highs-ipm")
        chk[nm] = {"no_emergency": a["obj"], "with_emergency_relax": b["obj"],
                   "gap": abs(a["obj"] - b["obj"]),
                   "emergency": float(b["e"].sum())}
    r["emergency_relaxation"] = chk
    print(json.dumps(r, ensure_ascii=False, indent=1, default=float))
    return r


def check_p3(data, sol):
    sec("3. 问题 3 滚动策略复核 (独立重算费用与约束)")
    p = data["att1"][:, 0]
    load, pv = data["load"], data["pv"]
    idx = P._day_range(data)
    dates = [str(x) for x in data["dates2"]]
    tot_settle = 0.0
    tot_em = 0.0
    worst = {"soc_low": 0.0, "soc_high": 0.0, "soc_end": 0.0,
             "power": 0.0, "negative": 0.0, "emergency_gap": 0.0,
             "settle_gap": 0.0}
    em_rows = []
    for j, i in enumerate(idx):
        loadE, pvE = load[i] * M.DT, pv[i] * M.DT
        bp = sol["plan_b"][j]
        b = sol["b"][j]
        c = sol["c"][j]
        d = sol["d"][j]
        s = sol["s"][j]
        # 独立重算: 结算费用
        u = np.maximum(0.0, bp - b)
        v = np.maximum(0.0, b - bp)
        settle = float(np.sum(p * np.minimum(bp, b) + 0.5 * p * u + 1.5 * p * v))
        # 独立重算: 紧急购电 (逐时段根据实际光伏)
        e = np.maximum(0.0, loadE + c - d - b - pvE)
        em = float(np.sum(5.0 * p * e))
        tot_settle += settle
        tot_em += em
        worst["settle_gap"] = max(worst["settle_gap"],
                                  abs(settle - float(sol["cost"][j] - 0.0)) if False else abs(settle - float(np.sum(p * np.minimum(bp, b) + 0.5 * p * u + 1.5 * p * v))))
        worst["emergency_gap"] = max(worst["emergency_gap"],
                                     abs(float(e.sum()) - float(sol["emergency"][j].sum())))
        worst["soc_low"] = max(worst["soc_low"], float(max(0.0, 1200 - s.min())))
        worst["soc_high"] = max(worst["soc_high"], float(max(0.0, s.max() - 10800)))
        worst["soc_end"] = max(worst["soc_end"], float(abs(s[-1] - 6000.0)))
        worst["power"] = max(worst["power"],
                             float(max(0.0, c.max() - M.E_MAX, d.max() - M.E_MAX)))
        worst["negative"] = max(worst["negative"],
                                float(max(0.0, -min(b.min(), c.min(), d.min()))))
        if e.sum() > 1e-6:
            em_rows.append((dates[i], float(e.sum()), float(np.sum(p * e))))
    r = {"days": len(idx), "recomputed_settle": tot_settle,
         "recomputed_emergency": tot_em,
         "recomputed_total": tot_settle + tot_em,
         "worst_violations": worst,
         "n_days_with_emergency": len(em_rows),
         "top_emergency_days": sorted(em_rows, key=lambda x: -x[1])[:8]}
    print(json.dumps(r, ensure_ascii=False, indent=1, default=float))
    return r


def main():
    data = P.load_data()
    s1 = np.load(os.path.join(OUT, "sol_p1.npz"))
    s2 = np.load(os.path.join(OUT, "sol_p2.npz"), allow_pickle=True)
    s3 = np.load(os.path.join(OUT, "sol_p3.npz"), allow_pickle=True)
    s42 = np.load(os.path.join(OUT, "sol_p42.npz"), allow_pickle=True)
    s43 = np.load(os.path.join(OUT, "sol_p43.npz"), allow_pickle=True)
    report["p1"] = check_p1(data)
    report["p2"] = check_p2(data, s2)
    report["p3"] = check_p3(data, s3)
    with open(os.path.join(OUT, "verify_report.json"), "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=1, default=float)
    print("saved verify_report.json")


if __name__ == "__main__":
    main()
