"""v2: 用随机规划 (SAA / 均值-CVaR / 鲁棒) 生成日前计划, 并用 v1 相同的滚动规则做样本外评价.

用法:
  python src/v2_run.py --objective saa   --exec A --out out/v2_saa_A.npz
  python src/v2_run.py --objective cvar  --lam 0.5 --alpha 0.9 ...
  python src/v2_run.py --objective robust --eps-mean 0.1 ...
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model as M
import pipeline as P
import v2_tree as V
from execution import run_interval_ruleA, run_interval_ruleB

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "out")


def simulate_plan(data, i, bp, c_init=None, d_init=None, price_mode="att1",
                  execution="A", rev_slots=(36, 72, 108), stages=(1, 2, 3)):
    """用给定的日前计划 bp 执行 v1 的滚动调整与结算, 返回实际发生费用.

    c_init/d_init: 日前计划对应的 0:00-6:00 充放电执行量 (来自同一优化);
                   为 None 时退化为用确定性 LP 由 bp 的日前预报反推。
    """
    p = (data["att1"][:, 0] if price_mode == "att1" else data["price_rt"][i])
    loadE = data["load"][i] * M.DT
    pv_act = data["pv"][i] * M.DT
    s_exec = np.empty(M.K)
    c_exec = np.zeros(M.K)
    d_exec = np.zeros(M.K)
    b_exec = bp.copy()
    # 阶段 0: 执行 0:00-6:00 (计划轨迹)
    f0 = P.forecast_energy(data, i, 0)
    if c_init is None:
        plan0 = M.solve_day(p, loadE, f0, s_init=6000.0, s_end=6000.0,
                            solver="highs-ds")
        c_exec[:36] = plan0["c"][:36]
        d_exec[:36] = plan0["d"][:36]
        s_exec[:36] = plan0["s"][:36]
    else:
        c_exec[:36] = np.asarray(c_init, float)[:36]
        d_exec[:36] = np.asarray(d_init, float)[:36]
        cur = 6000.0
        for k in range(36):
            cur = cur + M.ETA * c_exec[k] - d_exec[k] / M.ETA
            s_exec[k] = cur
    emerg = np.zeros(M.K)
    spill = np.zeros(M.K)
    em_cost = 0.0
    prev = 0
    for si, tau in enumerate(rev_slots):
        if execution == "A":
            sim = run_interval_ruleA(b_exec[prev:tau], c_exec[prev:tau],
                                     d_exec[prev:tau], p[prev:tau],
                                     loadE[prev:tau], pv_act[prev:tau],
                                     s_exec[prev - 1] if prev > 0 else 6000.0)
        else:
            sim = run_interval_ruleB(b_exec[prev:tau], c_exec[prev:tau],
                                     d_exec[prev:tau], p[prev:tau],
                                     loadE[prev:tau], pv_act[prev:tau],
                                     s_exec[prev - 1] if prev > 0 else 6000.0)
        emerg[prev:tau] = sim["e"]
        spill[prev:tau] = sim["spill"]
        c_exec[prev:tau] = sim["c"]
        d_exec[prev:tau] = sim["d"]
        s_exec[prev:tau] = sim["s"]
        em_cost += sim["emergency_cost"]
        f = P.forecast_energy(data, i, stages[si])
        sub = M.solve_day(p[tau:], loadE[tau:], f[tau:],
                          s_init=s_exec[tau - 1], s_end=6000.0,
                          plan=bp[tau:], solver="highs-ds")
        b_exec[tau:] = sub["b"]
        c_exec[tau:] = sub["c"]
        d_exec[tau:] = sub["d"]
        s_exec[tau:] = sub["s"]
        prev = tau
    if execution == "A":
        sim = run_interval_ruleA(b_exec[prev:], c_exec[prev:],
                                 d_exec[prev:], p[prev:], loadE[prev:],
                                 pv_act[prev:],
                                 s_exec[prev - 1] if prev > 0 else 6000.0)
    else:
        sim = run_interval_ruleB(b_exec[prev:], c_exec[prev:],
                                 d_exec[prev:], p[prev:], loadE[prev:],
                                 pv_act[prev:],
                                 s_exec[prev - 1] if prev > 0 else 6000.0)
    emerg[prev:] = sim["e"]
    spill[prev:] = sim["spill"]
    c_exec[prev:] = sim["c"]
    d_exec[prev:] = sim["d"]
    s_exec[prev:] = sim["s"]
    em_cost += sim["emergency_cost"]
    u = np.maximum(0.0, bp - b_exec)
    v = np.maximum(0.0, b_exec - bp)
    settle = float(np.sum(p * np.minimum(bp, b_exec) + 0.5 * p * u
                          + 1.5 * p * v))
    return {"total": settle + em_cost, "settle": settle,
            "emergency_cost": em_cost, "emergency": emerg,
            "spill": spill, "b": b_exec, "c": c_exec, "d": d_exec,
            "s": s_exec, "plan": bp}


def run(objective="saa", exec_rule="A", lam=0.5, alpha=0.9, eps_mean=0.0,
        n_scen=96, branches=(4, 3, 2), window=75, out=None, days=None,
        price_mode="att1", deg_cost=0.0):
    data = P.load_data()
    cache = V.make_cache(data)
    idx = P._day_range(data)
    if days is not None:
        idx = [i for i in idx if i in set(days)]
    rec = {"date": [], "cost": [], "settle": [], "em_cost": [],
           "em_energy": [], "obj_in": [], "mean_in": [], "cvar_in": [],
           "worst_in": [], "plan": [], "b": [], "c": [], "d": [], "s": [],
           "emergency": []}
    t0 = time.time()
    for cnt, i in enumerate(idx):
        scen = V.build_scenarios(data, i, window=window, n_scen=n_scen,
                                 cache=cache)
        tree = V.build_tree(scen, branches)
        p = (data["att1"][:, 0] if price_mode == "att1" else data["price_rt"][i])
        loadE = data["load"][i] * M.DT
        r = V.solve_tree(p, loadE, tree, scen, objective=objective, lam=lam,
                         alpha=alpha, eps_mean=eps_mean, deg_cost=deg_cost)
        bp = r["bP"]
        ev = simulate_plan(data, i, bp, price_mode=price_mode,
                           execution=exec_rule)
        rec["date"].append(str(data["dates2"][i]))
        rec["cost"].append(ev["total"])
        rec["settle"].append(ev["settle"])
        rec["em_cost"].append(ev["emergency_cost"])
        rec["em_energy"].append(float(ev["emergency"].sum()))
        rec["obj_in"].append(r["obj"])
        rec["mean_in"].append(r["mean_cost"])
        rec["cvar_in"].append(r["cvar"])
        rec["worst_in"].append(r["worst"])
        rec["plan"].append(bp)
        rec["b"].append(ev["b"])
        rec["c"].append(ev["c"])
        rec["d"].append(ev["d"])
        rec["s"].append(ev["s"])
        rec["emergency"].append(ev["emergency"])
        if cnt % 20 == 0:
            print(f"[{objective}/{exec_rule}] {cnt+1}/{len(idx)} "
                  f"{rec['date'][-1]} 费用 {ev['total']:,.0f} "
                  f"累计 {sum(rec['cost']):,.0f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    if out:
        np.savez_compressed(
            os.path.join(OUT, out),
            dates=np.array(rec["date"]),
            cost=np.array(rec["cost"]),
            settle=np.array(rec["settle"]),
            em_cost=np.array(rec["em_cost"]),
            em_energy=np.array(rec["em_energy"]),
            obj_in=np.array(rec["obj_in"]),
            mean_in=np.array(rec["mean_in"]),
            cvar_in=np.array(rec["cvar_in"]),
            worst_in=np.array(rec["worst_in"]),
            plan=np.array(rec["plan"]), b=np.array(rec["b"]),
            c=np.array(rec["c"]), d=np.array(rec["d"]),
            s=np.array(rec["s"]), emergency=np.array(rec["emergency"]))
        print("saved", out, flush=True)
    tot = float(np.sum(rec["cost"]))
    print(f"[{objective}/{exec_rule}] 全年(抽样 {len(idx)} 天) 总费用 {tot:,.2f} 元",
          flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objective", default="saa")
    ap.add_argument("--exec", dest="exec_rule", default="A")
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--alpha", type=float, default=0.9)
    ap.add_argument("--eps-mean", dest="eps_mean", type=float, default=0.0)
    ap.add_argument("--n-scen", dest="n_scen", type=int, default=96)
    ap.add_argument("--window", type=int, default=75)
    ap.add_argument("--price", default="att1")
    ap.add_argument("--deg", type=float, default=0.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--step", type=int, default=1)
    args = ap.parse_args()
    data = P.load_data()
    idx = P._day_range(data)
    days = idx[::args.step] if args.step > 1 else None
    run(objective=args.objective, exec_rule=args.exec_rule, lam=args.lam,
        alpha=args.alpha, eps_mean=args.eps_mean, n_scen=args.n_scen,
        window=args.window, out=args.out, days=days, price_mode=args.price,
        deg_cost=args.deg)


if __name__ == "__main__":
    main()
