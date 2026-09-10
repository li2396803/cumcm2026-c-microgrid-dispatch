"""v2: 全年滚动求解 (场景树策略 / MPC 闭环).

  python src/v2_year.py --mode tree --objective saa  --out v2_saa.npz
  python src/v2_year.py --mode tree --objective cvar --lam 0.5 --out v2_cvar.npz
  python src/v2_year.py --mode tree --objective robust --eps-mean .1 --out v2_robust.npz
  python src/v2_year.py --mode mpc  --objective saa  --step 2 --out v2_mpc.npz
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
import v2_policy as POL
from execution import run_interval_ruleA, run_interval_ruleB

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "out")
STAGE_SLOT = [0, 36, 72, 108, 144]


def build_subtree(data, i, cache, tau_m, branches, n_scen, window=75):
    """构造从第 tau_m 个决策点起的子场景树 (隐式: 只改 stage 起点)."""
    scen = V.build_scenarios(data, i, window=window, n_scen=n_scen,
                             cache=cache)
    stages = [STAGE_SLOT[m] for m in range(tau_m, 4)]
    tree = _build_tree_stages(scen, branches, stages, tau_m)
    return scen, tree


def _build_tree_stages(scen, branches, stages, tau_m):
    """与 V.build_tree 相同, 但阶段起点可自定 (用于 MPC 重优化)."""
    S = scen["act"].shape[0]
    nodes = [{"index": 0, "stage": tau_m, "parent": None, "children": [],
              "prob": 1.0, "scen": list(range(S)),
              "k0": stages[0], "k1": stages[1],
              "feat_lo": -np.inf, "feat_hi": np.inf}]
    prob = np.full(S, 1.0 / S)
    frontier = [nodes[0]]
    for step, nb in enumerate(branches):
        m = tau_m + step + 1                      # 使用的预报编号
        new_frontier = []
        for nd in frontier:
            ss = np.array(nd["scen"])
            feat = scen["fc"][m][ss][:, stages[step + 1]:].sum(axis=1)
            order = np.argsort(feat)
            for ch in np.array_split(order, nb):
                sub = ss[ch]
                fch = feat[ch]
                child = {"index": len(nodes), "stage": m, "parent": nd,
                         "children": [], "prob": float(prob[sub].sum()),
                         "scen": list(sub),
                         "k0": stages[step + 1],
                         "k1": stages[step + 2] if step + 2 < len(stages)
                         else 144,
                         "feat_lo": float(fch.min()), "feat_hi": float(fch.max()),
                         "parent_index": nd["index"]}
                nd["children"].append(child)
                nodes.append(child)
                new_frontier.append(child)
        frontier = new_frontier
    return {"nodes": nodes, "S": S}


def eval_mpc(data, i, cache, bp, objective="saa", branches=(4, 3, 2),
             n_scen=96, lam=0.5, alpha=0.9, eps_mean=0.0, deg_cost=0.0,
             buffering=True, window=75, p=None):
    """MPC: 每个决策点用最新信息与当前 SOC 重优化子问题, 段内允许储能实时平衡."""
    p = data["att1"][:, 0] if p is None else p
    loadE = data["load"][i] * M.DT
    pv_act = data["pv"][i] * M.DT
    b = np.zeros(M.K)
    c = np.zeros(M.K)
    d = np.zeros(M.K)
    emerg = np.zeros(M.K)
    cur = 6000.0
    for stage_m in range(4):
        k0, k1 = STAGE_SLOT[stage_m], STAGE_SLOT[stage_m + 1]
        nbr = tuple(branches[:3 - stage_m]) if stage_m < 3 else ()
        if len(nbr) == 0:
            # 最后一段: 直接用剩余 SOC 走到 6000 (确定性)
            sub = M.solve_day(np.zeros(k1 - k0) + p[k0:k1], loadE[k0:k1],
                              np.zeros(k1 - k0), s_init=cur, s_end=6000.0,
                              solver="highs-ds")
            bb, cc, dd = sub["b"], sub["c"], sub["d"]
        else:
            scen, tree = build_subtree(data, i, cache, stage_m, nbr, n_scen,
                                       window)
            # 子问题: 用最新信息与当前 SOC 重优化; 日前计划 bP 固定不可改
            r = V.solve_tree(p, loadE, tree, scen, objective=objective,
                             lam=lam, alpha=alpha, eps_mean=eps_mean,
                             deg_cost=deg_cost, bP_fixed=bp, s_init_root=cur)
            node0 = r["node_sol"][0]
            bb, cc, dd = node0["b"], node0["c"], node0["d"]
        # 段内执行
        if buffering:
            sim = run_interval_ruleB(bb, cc, dd, p[k0:k1], loadE[k0:k1],
                                     pv_act[k0:k1], cur)
            cc, dd = sim["c"], sim["d"]
        else:
            sim = run_interval_ruleA(bb, cc, dd, p[k0:k1], loadE[k0:k1],
                                     pv_act[k0:k1], cur)
        b[k0:k1] = bb
        c[k0:k1] = cc
        d[k0:k1] = dd
        emerg[k0:k1] = sim["e"]
        cur = sim["s"][-1]
    u = np.maximum(0.0, bp - b)
    v = np.maximum(0.0, b - bp)
    settle = float(np.sum(p * np.minimum(bp, b) + 0.5 * p * u + 1.5 * p * v))
    em = float(np.sum(5.0 * p * emerg))
    deg = float(deg_cost * np.sum(M.ETA * c + d / M.ETA))
    return {"total": settle + em + deg, "settle": settle,
            "emergency_cost": em, "em_energy": float(emerg.sum()),
            "deg_cost": deg, "b": b, "c": c, "d": d, "emergency": emerg,
            "soc_end": float(cur)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="tree")          # tree | mpc
    ap.add_argument("--objective", default="saa")      # saa | cvar | robust
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--alpha", type=float, default=0.9)
    ap.add_argument("--eps-mean", dest="eps_mean", type=float, default=0.0)
    ap.add_argument("--deg", type=float, default=0.0)
    ap.add_argument("--n-scen", dest="n_scen", type=int, default=96)
    ap.add_argument("--window", type=int, default=75)
    ap.add_argument("--branches", default="4,3,2")
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--buffer", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stale", type=int, default=0)
    ap.add_argument("--price", default="att1")
    args = ap.parse_args()
    br = tuple(int(v) for v in args.branches.split(","))
    data = P.load_data()
    cache = V.make_cache(data)
    idx = P._day_range(data)
    if args.step > 1:
        idx = idx[::args.step]
    p = data["att1"][:, 0]
    p_all = data["att1"][:, 0]
    rec = {k: [] for k in ["date", "cost", "settle", "em_cost", "em_energy",
                           "obj_in", "mean_in", "cvar_in", "worst_in"]}
    plans, bs, cs, ds, es = [], [], [], [], []
    t0 = time.time()
    for cnt, i in enumerate(idx):
        loadE = data["load"][i] * M.DT
        p = p_all if args.price == "att1" else data["price_rt"][i]
        if args.mode == "tree":
            scen = V.build_scenarios(data, i, window=args.window,
                                     n_scen=args.n_scen, cache=cache)
            if args.stale:
                scen = V.stale_scen(scen)
            tree = V.build_tree(scen, br)
            r = V.solve_tree(p, loadE, tree, scen, objective=args.objective,
                             lam=args.lam, alpha=args.alpha,
                             eps_mean=args.eps_mean, deg_cost=args.deg)
            ev = POL.evaluate_tree_policy(data, i, tree, r,
                                          deg_cost=args.deg,
                                          stale=bool(args.stale),
                                          price_mode=args.price)
            bp = r["bP"]
            obj_in = r["obj"]
            mean_in, cvar_in, worst_in = r["mean_cost"], r["cvar"], r["worst"]
        else:
            # 先用 SAA 树得到日前计划, 再按 MPC 执行
            scen = V.build_scenarios(data, i, window=args.window,
                                     n_scen=args.n_scen, cache=cache)
            tree = V.build_tree(scen, br)
            r = V.solve_tree(p, loadE, tree, scen, objective=args.objective,
                             lam=args.lam, alpha=args.alpha,
                             eps_mean=args.eps_mean, deg_cost=args.deg)
            bp = r["bP"]
            ev = eval_mpc(data, i, cache, bp, objective=args.objective,
                          branches=br, n_scen=args.n_scen, lam=args.lam,
                          alpha=args.alpha, eps_mean=args.eps_mean,
                          deg_cost=args.deg, buffering=bool(args.buffer),
                          window=args.window, p=p)
            obj_in, mean_in, cvar_in, worst_in = (r["obj"], r["mean_cost"],
                                                  r["cvar"], r["worst"])
        rec["date"].append(str(data["dates2"][i]))
        rec["cost"].append(ev["total"])
        rec["settle"].append(ev["settle"])
        rec["em_cost"].append(ev["emergency_cost"])
        rec["em_energy"].append(ev["em_energy"])
        rec["obj_in"].append(obj_in)
        rec["mean_in"].append(mean_in)
        rec["cvar_in"].append(cvar_in)
        rec["worst_in"].append(worst_in)
        plans.append(bp)
        bs.append(ev["b"])
        cs.append(ev["c"])
        ds.append(ev["d"])
        es.append(ev["emergency"])
        if cnt % 20 == 0:
            print(f"[{args.mode}/{args.objective}] {cnt+1}/{len(idx)} "
                  f"{rec['date'][-1]} {ev['total']:,.0f} "
                  f"累计 {sum(rec['cost']):,.0f} ({time.time()-t0:.0f}s)",
                  flush=True)
    np.savez_compressed(
        os.path.join(OUT, args.out),
        dates=np.array(rec["date"]), cost=np.array(rec["cost"]),
        settle=np.array(rec["settle"]), em_cost=np.array(rec["em_cost"]),
        em_energy=np.array(rec["em_energy"]), obj_in=np.array(rec["obj_in"]),
        mean_in=np.array(rec["mean_in"]), cvar_in=np.array(rec["cvar_in"]),
        worst_in=np.array(rec["worst_in"]), plan=np.array(plans),
        b=np.array(bs), c=np.array(cs), d=np.array(ds),
        emergency=np.array(es))
    tot = float(np.sum(rec["cost"]))
    em = float(np.sum(rec["em_energy"]))
    print(f"[{args.mode}/{args.objective}] n={len(idx)} 总费用 {tot:,.2f} 元, "
          f"紧急购电 {em:,.1f} kWh ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
