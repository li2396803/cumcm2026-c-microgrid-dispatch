"""v2 改进 1/2: 场景树生成 + 节点式多阶段随机规划 (SAA / 均值-CVaR / 最坏情况鲁棒).

场景生成: "相似日重采样 + 能量缩放"
  对目标日 d (0:00 预报 F0_d):
    候选相似日 d' ∈ 以 d 为中心的 ±W 天窗口 (排除 d 本身)
    缩放系数  s_d' = Σ_h F0_d(h) / Σ_h F0_d'(h)
    场景实际光伏  A_s = pv[d'] * s_d'
    场景第 m 次预报  Fm_s = A_s + (Fm_{d'} - pv_{d'}) * s_d'      (仅 h >= T_m)
  即保留相似日"预报误差形状"(云量过程)与误差相关性。

场景树: 在每个非叶节点内, 以"该时刻新发布的预报在剩余时段的总量"为特征
        (该特征在调整时刻可观测, 满足非预期性), 按分位数分箱得到子节点。

模型 (节点式):
  阶段 0 节点: 全天计划 bP (144) + 0:00-6:00 的执行量与储电量
  阶段 m>=1 节点: 本段 [T_m,T_{m+1}) 的 b,c,d,s 与偏差 u,v (b = bP - u + v)
  每个 (节点, 场景) 的紧急购电 e (段内实际光伏与计划不匹配)
  目标可切换: SAA(期望) / 均值-CVaR / 最坏情况(有限模糊集)
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix

import model as M
from pipeline import forecast_energy

K = M.K
DT = M.DT
ETA = M.ETA
E_MAX = M.E_MAX
S_MIN, S_MAX, S0 = 1200.0, 10800.0, 6000.0
STAGE_SLOT = [0, 36, 72, 108, 144]      # 各阶段起始时段


# ------------------------------------------------------------------ 场景 ----
def make_cache(data):
    """预计算: 附件3 预报按"小时电量(kWh)"存, 附件2 光伏按小时电量存."""
    fc = np.asarray(data["fc3"], float).reshape(365, 4, 24)   # kW·h/h
    pvE = np.asarray(data["pv"], float) * DT                  # kWh/slot
    pv_hour = pvE.reshape(365, 24, 6).sum(axis=2)             # kWh/h
    return {"fc_hour": fc, "pv_hour": pv_hour, "pvE": pvE}


def build_scenarios(data, day_row, window=75, n_scen=96, cache=None):
    """生成相似日场景集合 (逐小时构造后展开为 144 时段).

    返回 dict(act=(S,144) kWh/时段, fc=[(S,144) x4], donors)
    """
    if cache is None:
        cache = make_cache(data)
    fc = cache["fc_hour"]
    pv_hour = cache["pv_hour"]
    tot0 = fc[day_row, 0].sum() + 1e-9
    lo, hi = max(0, day_row - window), min(364, day_row + window)
    cand = np.concatenate([np.arange(lo, day_row), np.arange(day_row + 1, hi + 1)])
    tots = fc[cand, 0].sum(axis=1)
    order = np.argsort(np.abs(tots - tot0))[:n_scen]
    pick = cand[order]
    sc = tot0 / np.maximum(tots[order], 1e-9)                  # (S,)
    act_h = pv_hour[pick] * sc[:, None]                        # (S,24) kWh/h
    S = len(pick)
    fcs_h = [np.zeros((S, 24)) for _ in range(4)]
    for m, T in enumerate([0, 6, 12, 18]):
        # 附件3 第 m 行第 k 个值 = 从 T 起第 k 小时(即时钟 T+k-1 时)的预报,
        # 因此目标小时 h (h>=T) 对应索引 k-1 = h-T。
        err_h = np.zeros((S, 24))
        for h in range(T, 24):
            err_h[:, h] = fc[pick, m, h - T] - pv_hour[pick, h]
        fcs_h[m][:, T:] = act_h[:, T:] + err_h[:, T:] * sc[:, None]
        fcs_h[m] = np.maximum(fcs_h[m], 0.0)
    act = np.repeat(act_h, 6, axis=1) * DT
    fcs = [np.repeat(f, 6, axis=1) * DT for f in fcs_h]
    return {"act": act, "fc": fcs, "donors": pick, "scale": sc}


def stale_scen(scen):
    """把场景集中 6:00/12:00/18:00 预报替换为首日 0:00 预报 (模拟不使用更新)."""
    return {"act": scen["act"], "donors": scen["donors"],
            "scale": scen["scale"],
            "fc": [scen["fc"][0].copy() for _ in range(4)]}


def _fc_energy(data, day_row, stage):
    """附件 3 第 stage 个预报展开成 144 时段电量 (kWh)."""
    row = data["fc3"][day_row * 4 + stage]
    start = [0, 6, 12, 18][stage]
    out = np.zeros(K)
    for k in range(1, 25):
        h0 = start + k - 1
        if h0 >= 24:
            break
        out[h0 * 6:(h0 + 1) * 6] = row[k - 1]
    return out * DT


# ------------------------------------------------------------------- 树 ----
def build_tree(scen, branches=(4, 3, 2)):
    """按"剩余时段预报总量"分位数构造场景树."""
    S = scen["act"].shape[0]
    nodes = [{"index": 0, "stage": 0, "parent": None, "children": [],
              "prob": 1.0, "scen": list(range(S)), "k0": 0, "k1": 36,
              "feat_lo": -np.inf, "feat_hi": np.inf}]
    prob = np.full(S, 1.0 / S)
    frontier = [nodes[0]]
    for m, nb in enumerate(branches, start=1):
        new_frontier = []
        for nd in frontier:
            ss = np.array(nd["scen"])
            # 特征: 该阶段新预报在剩余时段的总量 (可观测)
            feat = scen["fc"][m][ss][:, STAGE_SLOT[m]:].sum(axis=1)
            order = np.argsort(feat)
            chunks = np.array_split(order, nb)
            for ch in chunks:
                sub = ss[ch]
                feat_ch = feat[ch]
                child = {"index": len(nodes), "stage": m, "parent": nd,
                         "children": [],
                         "prob": float(prob[sub].sum()), "scen": list(sub),
                         "k0": STAGE_SLOT[m], "k1": STAGE_SLOT[m + 1],
                         "feat_lo": float(feat_ch.min()),
                         "feat_hi": float(feat_ch.max())}
                child["parent_index"] = nd["index"]
                nd["children"].append(child)
                nodes.append(child)
                new_frontier.append(child)
        frontier = new_frontier
    return {"nodes": nodes, "S": S}


# ---------------------------------------------------------------- 节点 LP ----
def solve_tree(price, loadE, tree, scen, objective="saa", lam=0.5, alpha=0.9,
               deg_cost=0.0, eps_mean=0.0, solver="highs-ds",
               settle_alpha=0.5, b_max=np.inf, bP_fixed=None,
               s_init_root=S0):

    """节点式多阶段随机规划, 返回根节点计划与各场景费用."""
    from scipy.sparse import csr_matrix
    nodes = tree["nodes"]
    S = tree["S"]
    act = scen["act"]
    pi = np.asarray(price, float)
    nid = {id(nd): i for i, nd in enumerate(nodes)}

    idx, nv = {}, 0

    def add(name, size):
        nonlocal nv
        idx[name] = slice(nv, nv + size)
        nv += size

    add("bP", K)
    for ni, nd in enumerate(nodes):
        L = nd["k1"] - nd["k0"]
        for pre in ("b", "c", "d", "s"):
            add(f"{pre}{ni}", L)
        if nd["stage"] >= 1:
            add(f"u{ni}", L)
            add(f"v{ni}", L)
        for s in nd["scen"]:
            add(f"e{ni}_{s}", L)
    prob = np.zeros(S)
    for nd in nodes:
        for s in nd["scen"]:
            prob[s] = nd["prob"] / len(nd["scen"])
    prob /= prob.sum()

    if objective == "cvar":
        add("eta", 1)
        add("z", S)
    elif objective == "robust":
        add("z", 1)
    nvar = nv

    # ---------------- 目标 ----------------
    cobj = np.zeros(nvar)
    for ni, nd in enumerate(nodes):
        pk = pi[nd["k0"]:nd["k1"]]
        w = nd["prob"]
        cobj[idx[f"b{ni}"]] += pk * w
        if nd["stage"] >= 1:
            cobj[idx[f"u{ni}"]] += settle_alpha * pk * w
            cobj[idx[f"v{ni}"]] += 0.5 * pk * w
        if deg_cost:
            cobj[idx[f"c{ni}"]] += deg_cost * ETA
            cobj[idx[f"d{ni}"]] += deg_cost / ETA
        for s in nd["scen"]:
            cobj[idx[f"e{ni}_{s}"]] += 5.0 * pk * (nd["prob"] / len(nd["scen"]))
    if objective == "cvar":
        cobj[idx["eta"]] = lam
        cobj[idx["z"].start:idx["z"].start + S] = lam * prob / (1.0 - alpha)
    elif objective == "robust":
        cobj[idx["z"]] = 1.0
        # 加一点期望项以消除并列 (默认 0)
        if eps_mean:
            cobj[idx["z"]] = 1.0 - eps_mean
            for ni, nd in enumerate(nodes):
                cobj[idx[f"b{ni}"]] += eps_mean * pi[nd["k0"]:nd["k1"]] * nd["prob"]
                if nd["stage"] >= 1:
                    cobj[idx[f"u{ni}"]] += eps_mean * 0.5 * pi[nd["k0"]:nd["k1"]] * nd["prob"]
                    cobj[idx[f"v{ni}"]] += eps_mean * 0.5 * pi[nd["k0"]:nd["k1"]] * nd["prob"]
                for s in nd["scen"]:
                    cobj[idx[f"e{ni}_{s}"]] += (eps_mean * 5.0
                                                * pi[nd["k0"]:nd["k1"]]
                                                * nd["prob"] / len(nd["scen"]))

    # ---------------- 约束收集 ----------------
    er, ec, ev, eb = [], [], [], []
    ur, uc, uv, ub_ = [], [], [], []
    n_eq = n_ub = 0

    def eq(entries, rhs):
        nonlocal n_eq
        for j, v in entries:
            er.append(n_eq)
            ec.append(j)
            ev.append(v)
        eb.append(rhs)
        n_eq += 1

    def ub(entries, rhs):
        nonlocal n_ub
        for j, v in entries:
            ur.append(n_ub)
            uc.append(j)
            uv.append(v)
        ub_.append(rhs)
        n_ub += 1

    for ni, nd in enumerate(nodes):
        L = nd["k1"] - nd["k0"]
        for j in range(L):
            ent = [(idx[f"s{ni}"].start + j, 1.0),
                   (idx[f"c{ni}"].start + j, -ETA),
                   (idx[f"d{ni}"].start + j, 1.0 / ETA)]
            if j > 0:
                ent.append((idx[f"s{ni}"].start + j - 1, -1.0))
                rhs = 0.0
            elif nd["parent"] is None:
                rhs = s_init_root
            else:
                par = nd["parent"]
                pni = nid[id(par)]
                ent.append((idx[f"s{pni}"].start
                            + (par["k1"] - par["k0"]) - 1, -1.0))
                rhs = 0.0
            eq(ent, rhs)
        if nd["stage"] >= 1:
            for j in range(L):
                k = nd["k0"] + j
                eq([(idx[f"b{ni}"].start + j, 1.0),
                    (idx[f"u{ni}"].start + j, 1.0),
                    (idx[f"v{ni}"].start + j, -1.0),
                    (idx["bP"].start + k, -1.0)], 0.0)
        # 平衡: -b - d + c - e <= act - load
        for s in nd["scen"]:
            for j in range(L):
                k = nd["k0"] + j
                ub([(idx[f"b{ni}"].start + j, -1.0),
                    (idx[f"d{ni}"].start + j, -1.0),
                    (idx[f"c{ni}"].start + j, 1.0),
                    (idx[f"e{ni}_{s}"].start + j, -1.0)], act[s, k] - loadE[k])
        if nd["stage"] >= 1:
            for j in range(L):
                k = nd["k0"] + j
                ub([(idx[f"u{ni}"].start + j, 1.0),
                    (idx["bP"].start + k, -1.0)], 0.0)
    # 根节点: 执行购电量 = 计划量 (根节点所在段)
    n0 = nodes[0]
    for j in range(n0["k1"] - n0["k0"]):
        eq([(idx["bP"].start + n0["k0"] + j, 1.0),
            (idx["b0"].start + j, -1.0)], 0.0)

    # 场景费用表达式 C_s
    scen_cost = []
    for s in range(S):
        ent = []
        for ni, nd in enumerate(nodes):
            if s not in nd["scen"]:
                continue
            pk = pi[nd["k0"]:nd["k1"]]
            L = nd["k1"] - nd["k0"]
            for j in range(L):
                ent.append((idx[f"b{ni}"].start + j, pk[j]))
                if nd["stage"] >= 1:
                    ent.append((idx[f"u{ni}"].start + j, 0.5 * pk[j]))
                    ent.append((idx[f"v{ni}"].start + j, 0.5 * pk[j]))
                ent.append((idx[f"e{ni}_{s}"].start + j, 5.0 * pk[j]))
        scen_cost.append(ent)

    if objective == "cvar":
        for s in range(S):
            ub(scen_cost[s] + [(idx["eta"].start, -1.0),
                               (idx["z"].start + s, -1.0)], 0.0)
    elif objective == "robust":
        for s in range(S):
            ub(scen_cost[s] + [(idx["z"].start, -1.0)], 0.0)

    # ---------------- 边界 ----------------
    lb = np.zeros(nvar)
    ubc = np.full(nvar, np.inf)
    for ni, nd in enumerate(nodes):
        ubc[idx[f"c{ni}"]] = E_MAX
        ubc[idx[f"d{ni}"]] = E_MAX
        lb[idx[f"s{ni}"]] = S_MIN
        ubc[idx[f"s{ni}"]] = S_MAX
        if np.isfinite(b_max):
            ubc[idx[f"b{ni}"]] = b_max
        if nd["stage"] == 3:
            last = idx[f"s{ni}"].start + (nd["k1"] - nd["k0"]) - 1
            lb[last] = S0
            ubc[last] = S0
    if np.isfinite(b_max):
        ubc[idx["bP"]] = b_max
    if bP_fixed is not None:
        lb[idx["bP"]] = np.asarray(bP_fixed, float)
        ubc[idx["bP"]] = np.asarray(bP_fixed, float)
    if objective == "cvar":
        lb[idx["eta"]] = -np.inf
    A_eq = csr_matrix((ev, (er, ec)), shape=(n_eq, nvar))
    A_ub = csr_matrix((uv, (ur, uc)), shape=(n_ub, nvar))
    res = linprog(cobj, A_ub=A_ub, b_ub=np.array(ub_), A_eq=A_eq,
                  b_eq=np.array(eb), bounds=list(zip(lb, ubc)), method=solver)
    if not res.success:
        raise RuntimeError("tree LP failed: " + str(res.message))
    x = res.x
    out = {"obj": float(res.fun), "bP": x[idx["bP"]],
           "c0": x[idx["c0"]], "d0": x[idx["d0"]], "s0": x[idx["s0"]],
           "cost": np.array([sum(coef * x[j] for j, coef in scen_cost[s])
                             for s in range(S)]),
           "prob": prob, "status": int(res.status)}
    out["node_sol"] = []
    for ni, nd in enumerate(nodes):
        L = nd["k1"] - nd["k0"]
        out["node_sol"].append({
            "stage": nd["stage"],
            "b": x[idx[f"b{ni}"]], "c": x[idx[f"c{ni}"]],
            "d": x[idx[f"d{ni}"]], "s": x[idx[f"s{ni}"]],
            "k0": nd["k0"], "k1": nd["k1"],
            "feat_lo": float(nd.get("feat_lo", -np.inf)),
            "feat_hi": float(nd.get("feat_hi", np.inf)),
            "parent": nd.get("parent_index", -1),
            "scen": list(nd["scen"])})
    out["mean_cost"] = float(out["cost"] @ prob)
    q = np.quantile(out["cost"], alpha, method="inverted_cdf")
    tail = out["cost"][out["cost"] > q]
    out["cvar"] = float(tail.mean()) if len(tail) else float(out["cost"].max())
    out["worst"] = float(out["cost"].max())
    out["lp_debug"] = {"cobj": cobj, "idx": {k: (v.start, v.stop)
                                             for k, v in idx.items()},
                       "A_ub": A_ub, "b_ub": np.array(ub_),
                       "A_eq": A_eq, "b_eq": np.array(eb), "x": x}
    return out
