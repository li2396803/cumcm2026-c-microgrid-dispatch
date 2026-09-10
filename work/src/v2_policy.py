"""v2: 场景树策略的样本外评价 (按实际预报特征逐级选节点, 执行该节点的决策)."""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model as M
import pipeline as P


def route(tree, res, feats):
    """按特征 feats[stage] 逐级选择节点, 返回每个阶段选中的节点下标."""
    nodes = tree["nodes"]
    chosen = [0]
    nd = nodes[0]
    for m in range(1, 4):
        f = feats[m]
        best, best_d = None, np.inf
        for ch in nd["children"]:
            lo, hi = ch["feat_lo"], ch["feat_hi"]
            if f < lo:
                dd = lo - f
            elif f > hi:
                dd = f - hi
            else:
                dd = 0.0
            if dd < best_d:
                best_d, best = dd, ch
        nd = best
        chosen.append(nd["index"])
    return chosen


def evaluate_tree_policy(data, i, tree, res, price_mode="att1", deg_cost=0.0,
                         stale=False):
    """把实际日按特征路由到树中, 执行各节点决策并结算实际费用."""
    nodes = tree["nodes"]
    p = data["att1"][:, 0] if price_mode == "att1" else data["price_rt"][i]
    loadE = data["load"][i] * M.DT
    pv_act = data["pv"][i] * M.DT
    feats = {m: float(P.forecast_energy(data, i, 0 if stale else m)[
        [0, 36, 72, 108][m]:].sum()) for m in range(4)}
    chosen = route(tree, res, feats)
    bP = res["bP"]
    b = np.zeros(M.K)
    c = np.zeros(M.K)
    d = np.zeros(M.K)
    for m, ni in enumerate(chosen):
        nd = nodes[ni]
        sol = res["node_sol"][ni]
        k0, k1 = nd["k0"], nd["k1"]
        b[k0:k1] = sol["b"]
        c[k0:k1] = sol["c"]
        d[k0:k1] = sol["d"]
    e = np.maximum(0.0, loadE + c - d - b - pv_act)
    u = np.maximum(0.0, bP - b)
    v = np.maximum(0.0, b - bP)
    settle = float(np.sum(p * np.minimum(bP, b) + 0.5 * p * u + 1.5 * p * v))
    em = float(np.sum(5.0 * p * e))
    deg = float(deg_cost * np.sum(M.ETA * c + d / M.ETA))
    return {"total": settle + em + deg, "settle": settle,
            "emergency_cost": em, "emergency": e,
            "em_energy": float(e.sum()), "deg_cost": deg,
            "b": b, "c": c, "d": d, "plan": bP, "nodes": chosen,
            "feats": [feats[m] for m in range(4)]}
