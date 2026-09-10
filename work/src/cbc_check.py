"""用 PuLP + CBC (与 HiGHS 完全独立的求解器) 复算导出的 LP 实例.

运行方式: /tmp/venvcumcm/bin/python src/cbc_check.py
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import pulp

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "out")
CASE_DIR = os.path.join(OUT, "lp_cases")

ETA = 0.9
EMAX = 5000.0 / 6.0
SMIN, SMAX = 1200.0, 10800.0


def solve(case):
    p = case["p"]
    loadE = case["loadE"]
    pvE = case["pvE"]
    s_init = float(case["s_init"][0])
    s_end = float(case["s_end"][0])
    plan = case["plan"] if "plan" in case else None
    fixed = case["fixed"] if "fixed" in case else None
    n = len(p)
    prob = pulp.LpProblem("m", pulp.LpMinimize)
    b = [pulp.LpVariable(f"b{k}", lowBound=0) for k in range(n)]
    c = [pulp.LpVariable(f"c{k}", lowBound=0, upBound=EMAX) for k in range(n)]
    d = [pulp.LpVariable(f"d{k}", lowBound=0, upBound=EMAX) for k in range(n)]
    s = [pulp.LpVariable(f"s{k}", lowBound=SMIN, upBound=SMAX) for k in range(n)]
    obj = pulp.lpSum(p[k] * b[k] for k in range(n))
    if plan is not None:
        u = [pulp.LpVariable(f"u{k}", lowBound=0, upBound=max(plan[k], 0.0))
             for k in range(n)]
        v = [pulp.LpVariable(f"v{k}", lowBound=0) for k in range(n)]
        obj += pulp.lpSum(0.5 * p[k] * u[k] + 1.5 * p[k] * v[k]
                          for k in range(n))
        for k in range(n):
            prob += b[k] + u[k] - v[k] == plan[k]
        if fixed is not None:
            for k in range(n):
                if fixed[k] > 0.5:
                    prob += u[k] == 0
                    prob += v[k] == 0
    prob += obj
    for k in range(n):
        prev = s_init if k == 0 else s[k - 1]
        prob += s[k] == prev + ETA * c[k] - d[k] / ETA
        prob += b[k] + d[k] + pvE[k] >= loadE[k] + c[k]
    prob += s[n - 1] == s_end
    prob.solve(pulp.PULP_CBC_CMD(msg=0, threads=1))
    ok = pulp.LpStatus[prob.status] == "Optimal"
    return {
        "status": pulp.LpStatus[prob.status],
        "obj": float(pulp.value(prob.objective)) if ok else None,
        "b_sum": float(sum(x.value() for x in b)) if ok else None,
        "c_sum": float(sum(x.value() for x in c)) if ok else None,
        "d_sum": float(sum(x.value() for x in d)) if ok else None,
    }


def main():
    out = {}
    for f in sorted(glob.glob(os.path.join(CASE_DIR, "*.npz"))):
        name = os.path.basename(f)[:-4]
        case = np.load(f, allow_pickle=True)
        out[name] = solve(case)
        print(name, out[name], flush=True)
    with open(os.path.join(OUT, "cbc_results.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
