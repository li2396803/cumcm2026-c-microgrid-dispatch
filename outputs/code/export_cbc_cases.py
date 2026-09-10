"""导出用于 CBC 独立求解器交叉验证的 LP 实例."""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline as P
import model as M

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "out")
CASE_DIR = os.path.join(OUT, "lp_cases")


def save_case(name, p, loadE, pvE, s_init, s_end, plan=None, fixed=None):
    d = {"p": p, "loadE": loadE, "pvE": pvE,
         "s_init": np.array([s_init]), "s_end": np.array([s_end])}
    if plan is not None:
        d["plan"] = plan
    if fixed is not None:
        d["fixed"] = fixed.astype(float)
    np.savez(os.path.join(CASE_DIR, name + ".npz"), **d)


def main():
    os.makedirs(CASE_DIR, exist_ok=True)
    data = P.load_data()
    a1 = data["att1"]
    p_fix = a1[:, 0]
    dates = [str(x) for x in data["dates2"]]

    # P1
    p, loadE, pvE = a1[:, 0], a1[:, 1] * M.DT, a1[:, 2] * M.DT
    save_case("p1_day", p, loadE, pvE, 6000.0, 6000.0)

    # P2: 指定日期
    for name in ["2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21"]:
        i = dates.index(name)
        save_case("p2_" + name, p_fix, data["load"][i] * M.DT,
                  data["pv"][i] * M.DT, 6000.0, 6000.0)
        save_case("p42_" + name, data["price_rt"][i],
                  data["load"][i] * M.DT, data["pv"][i] * M.DT, 6000.0, 6000.0)

    # P3: 0:00 计划 + 各调整时刻子问题 (2025-03-20)
    i = dates.index("2025-03-20")
    loadE = data["load"][i] * M.DT
    pv_act = data["pv"][i] * M.DT
    f0 = P.forecast_energy(data, i, 0)
    plan = P.plan_day(p_fix, loadE, f0, 6000.0, 6000.0)
    bp, cp, dp = plan["b"], plan["c"], plan["d"]
    save_case("p3_plan_" + "2025-03-20", p_fix, loadE, f0, 6000.0, 6000.0)
    s_exec = plan["s"].copy()
    c_exec, d_exec, b_exec = cp.copy(), dp.copy(), bp.copy()
    for si, tau in enumerate([36, 72, 108], start=1):
        prev = [36, 72, 108][si - 1] if si > 0 else 0
        prev = [0, 36, 72][si - 1]
        from execution import run_interval_ruleA
        sim = run_interval_ruleA(b_exec[prev:tau], c_exec[prev:tau],
                                 d_exec[prev:tau], p_fix[prev:tau],
                                 loadE[prev:tau], pv_act[prev:tau],
                                 s_exec[prev - 1] if prev > 0 else 6000.0)
        s_exec[prev:tau] = sim["s"]
        f = P.forecast_energy(data, i, si)
        save_case(f"p3_stage{si}_2025-03-20", p_fix[tau:], loadE[tau:],
                  f[tau:], s_exec[tau - 1], 6000.0, plan=bp[tau:])
        sub = M.solve_day(p_fix[tau:], loadE[tau:], f[tau:],
                          s_init=s_exec[tau - 1], s_end=6000.0,
                          plan=bp[tau:], solver="highs-ds")
        b_exec[tau:], c_exec[tau:], d_exec[tau:] = sub["b"], sub["c"], sub["d"]
        s_exec[tau:] = sub["s"]
    print("cases:", sorted(os.listdir(CASE_DIR)))


if __name__ == "__main__":
    main()
