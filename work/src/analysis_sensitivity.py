"""分析三: 关键建模假设的灵敏度与模型变体对比.

(1) 预报-实测时间对齐 (0 / +20 / +30 / -10 分钟);
(2) 结算规则口径 (边际结算 vs 全额计划+违约金);
(3) 全年滚动 (允许跨日储电量结转) 与逐日闭环约束的差异;
(4) 初始储电量灵敏度。
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model as M
import pipeline as P

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "out")


def alignment(data):
    res = {}
    for shift, name in [(0, "0 (基准)"), (2, "+20 min"), (3, "+30 min"),
                        (-1, "-10 min")]:
        r = P.problem3(data, "att1", fc_shift=shift)
        tot = sum(x["total"] for x in r)
        em = sum(x["cost_emergency"] for x in r)
        res[name] = {"shift_slots": shift, "total": tot, "emergency": em}
        print(f"[时间对齐] {name:10s} 总费用 {tot:,.0f} 元 (紧急 {em:,.0f})",
              flush=True)
    return res


def settlement_variant(data, hedge=0.0):
    """口径 A: 违约部分按计划全额计费 + 50% 违约金 (取-or-付)."""
    a1 = data["att1"]
    p_fix = a1[:, 0]
    idx = P._day_range(data)
    load, pv = data["load"], data["pv"]
    tot = 0.0
    em_total = 0.0
    for i in idx:
        loadE = load[i] * M.DT
        pv_act = pv[i] * M.DT
        f0 = P.forecast_energy(data, i, 0) * (1 - hedge)
        plan = P.plan_day(p_fix, loadE, f0, 6000, 6000)
        bp, cp, dp = plan["b"], plan["c"], plan["d"]
        s_exec = plan["s"].copy()
        b_exec, c_exec, d_exec = bp.copy(), cp.copy(), dp.copy()
        em = np.zeros(M.K)
        em_cost = 0.0
        for si, tau in enumerate([0, 36, 72, 108]):
            if si > 0:
                prev = [0, 36, 72][si - 1]
                e = np.maximum(0.0, loadE[prev:tau] + c_exec[prev:tau]
                               - d_exec[prev:tau] - b_exec[prev:tau]
                               - pv_act[prev:tau])
                em[prev:tau] = e
                em_cost += float(np.sum(5.0 * p_fix[prev:tau] * e))
            if tau == 0:
                continue
            f = P.forecast_energy(data, i, si) * (1 - hedge)
            sub = M.solve_day(p_fix[tau:], loadE[tau:], f[tau:],
                              s_init=s_exec[tau - 1], s_end=6000.0,
                              plan=bp[tau:], solver="highs-ds")
            b_exec[tau:], c_exec[tau:], d_exec[tau:] = (sub["b"], sub["c"],
                                                        sub["d"])
            s_exec[tau:] = sub["s"]
        e = np.maximum(0.0, loadE + c_exec - d_exec - b_exec - pv_act)
        em_cost += float(np.sum(5.0 * p_fix * e))
        # 口径 A: 计划费用全额 + 削减部分 50% 违约金 + 增加部分 150%
        u = np.maximum(0.0, bp - b_exec)
        v = np.maximum(0.0, b_exec - bp)
        cost = float(np.sum(p_fix * bp) + np.sum(0.5 * p_fix * u)
                     + np.sum(1.5 * p_fix * v))
        tot += cost + em_cost
    return {"total": tot}


def year_horizon(data, price_mode="att1"):
    """全年单一 LP: 储电量可跨日结转, 年末回到 6000 (不设逐日闭环)."""
    load, pv = data["load"], data["pv"]
    idx = P._day_range(data)
    n_day = len(idx)
    n = n_day * M.K
    p = np.zeros(n)
    loadE = np.zeros(n)
    pvE = np.zeros(n)
    for j, i in enumerate(idx):
        p[j * M.K:(j + 1) * M.K] = (data["att1"][:, 0] if price_mode == "att1"
                                    else data["price_rt"][i])
        loadE[j * M.K:(j + 1) * M.K] = load[i] * M.DT
        pvE[j * M.K:(j + 1) * M.K] = pv[i] * M.DT
    nv = 4 * n
    cobj = np.zeros(nv)
    cobj[0:n] = p
    Aeq = lil_matrix((n, nv))
    for k in range(n):
        Aeq[k, 3 * n + k] = 1.0
        Aeq[k, n + k] = -M.ETA
        Aeq[k, 2 * n + k] = 1.0 / M.ETA
        if k > 0:
            Aeq[k, 3 * n + k - 1] = -1.0
    beq = np.zeros(n)
    beq[0] = 6000.0
    A_ub = lil_matrix((n, nv))
    b_ub = np.zeros(n)
    for k in range(n):
        A_ub[k, k] = -1.0
        A_ub[k, 2 * n + k] = -1.0
        A_ub[k, n + k] = 1.0
        b_ub[k] = pvE[k] - loadE[k]
    lb = np.zeros(nv)
    ub = np.full(nv, np.inf)
    ub[n:2 * n] = M.E_MAX
    ub[2 * n:3 * n] = M.E_MAX
    lb[3 * n:4 * n] = 1200.0
    ub[3 * n:4 * n] = 10800.0
    ub[3 * n + n - 1] = 6000.0
    lb[3 * n + n - 1] = 6000.0
    res = linprog(cobj, A_ub=A_ub.tocsc(), b_ub=b_ub, A_eq=Aeq.tocsc(),
                  b_eq=beq, bounds=list(zip(lb, ub)), method="highs")
    return {"success": bool(res.success), "obj": float(res.fun) if
            res.success else None, "message": res.message}


def main():
    data = P.load_data()
    res = {}
    res["alignment"] = alignment(data)
    with open(os.path.join(OUT, "analysis_sensitivity.json"), "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
