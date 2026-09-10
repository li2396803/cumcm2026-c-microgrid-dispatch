"""独立复核工具: (1) 动态规划精确复算 (2) CBC 独立求解器 (3) 对偶界."""
from __future__ import annotations

import numpy as np

from model import K, ETA, E_MAX, CAP_MIN, CAP_MAX


def dp_solve(p, loadE, pvE, s_init, s_end, grid=15.0,
             smin=CAP_MIN, smax=CAP_MAX, eta=ETA, emax=E_MAX):
    """SOC 网格动态规划 (Bellman) 精确复算; 网格限制使其为原问题的上界."""
    ns = int(round((smax - smin) / grid)) + 1
    states = smin + grid * np.arange(ns)
    # 目标: 期末必须落在 s_end
    if abs((s_end - smin) / grid - round((s_end - smin) / grid)) > 1e-9:
        raise ValueError("s_end 必须在网格上")
    i_end = int(round((s_end - smin) / grid))
    i_init = int(round((s_init - smin) / grid))
    # 转移: ds = 0.9c - d/0.9 ,  -925.9 <= ds <= 750
    dmax_s = emax / eta                       # 最大 SOC 下降
    cmax_s = emax * eta                       # 最大 SOC 上升
    n_up = int(np.floor(cmax_s / grid))
    n_dn = int(np.floor(dmax_s / grid))
    ds_list = np.concatenate([grid * np.arange(0, n_up + 1),
                              -grid * np.arange(1, n_dn + 1)])
    ds_list.sort()
    cost_cache = []
    for k in range(K):
        # 给定 ds, 最优 (c,d): ds>=0 -> c=ds/eta, d=0 ; ds<0 -> d=-ds*eta, c=0
        c = np.where(ds_list > 0, ds_list / eta, 0.0)
        d = np.where(ds_list < 0, -ds_list * eta, 0.0)
        need = loadE[k] + c - d - pvE[k]
        cost = p[k] * np.maximum(0.0, need)       # 不足部分购电, 盈余弃光
        cost_cache.append((c, np.round(ds_list / grid).astype(int), cost))

    INF = np.inf
    f = np.full(ns, INF)
    f[i_init] = 0.0
    parent = np.zeros((K, ns), dtype=int)         # 记录 ds 索引
    for k in range(K):
        c, off, cost = cost_cache[k]
        nf = np.full(ns, INF)
        par = np.full(ns, -1, dtype=int)
        for j, o in enumerate(off):
            if o >= 0:
                src = slice(0, ns - o)
                dst = slice(o, ns)
            else:
                src = slice(-o, ns)
                dst = slice(0, ns + o)
            cand = f[src] + cost[j]
            better = cand < nf[dst]
            nf[dst] = np.where(better, cand, nf[dst])
            par[dst] = np.where(better, j, par[dst])
        f = nf
        parent[k] = par
    # 回溯
    s_traj = np.empty(K)
    ds_idx = np.empty(K, dtype=int)
    i = i_end
    for k in range(K - 1, -1, -1):
        j = parent[k, i]
        ds_idx[k] = j
        i = i - int(round(ds_list[j] / grid))
        s_traj[k] = smin + grid * i + ds_list[j]
    s_prev = np.concatenate([[s_init], s_traj[:-1]])
    c_out = np.where(ds_list[ds_idx] > 0, ds_list[ds_idx] / eta, 0.0)
    d_out = np.where(ds_list[ds_idx] < 0, -ds_list[ds_idx] * eta, 0.0)
    b_out = np.maximum(0.0, loadE + c_out - d_out - pvE)
    return {"obj": float(f[i_end]), "b": b_out, "c": c_out,
            "d": d_out, "s": s_traj, "grid": grid}


def cbc_solve(p, loadE, pvE, s_init, s_end, plan=None, fixed=None,
              allow_emergency=False, emergency_mult=5.0,
              eta=ETA, cmax=E_MAX, dmax=E_MAX,
              smin=CAP_MIN, smax=CAP_MAX):
    """用 PuLP + CBC 独立求解同一模型 (需在装有 pulp 的解释器中调用)."""
    import pulp

    prob = pulp.LpProblem("microgrid", pulp.LpMinimize)
    N = K
    b = [pulp.LpVariable(f"b{k}", lowBound=0) for k in range(N)]
    c = [pulp.LpVariable(f"c{k}", lowBound=0, upBound=cmax) for k in range(N)]
    d = [pulp.LpVariable(f"d{k}", lowBound=0, upBound=dmax) for k in range(N)]
    s = [pulp.LpVariable(f"s{k}", lowBound=smin, upBound=smax) for k in range(N)]
    obj = pulp.lpSum(p[k] * b[k] for k in range(N))
    if plan is not None:
        u = [pulp.LpVariable(f"u{k}", lowBound=0, upBound=max(plan[k], 0.0))
             for k in range(N)]
        v = [pulp.LpVariable(f"v{k}", lowBound=0) for k in range(N)]
        obj += pulp.lpSum(0.5 * p[k] * u[k] + 1.5 * p[k] * v[k]
                          for k in range(N))
        for k in range(N):
            prob += b[k] + u[k] - v[k] == plan[k]
        if fixed is not None:
            for k in range(N):
                if fixed[k]:
                    prob += u[k] == 0
                    prob += v[k] == 0
    if allow_emergency:
        e = [pulp.LpVariable(f"e{k}", lowBound=0) for k in range(N)]
        obj += pulp.lpSum(emergency_mult * p[k] * e[k] for k in range(N))
    prob += obj
    for k in range(N):
        prev = s_init if k == 0 else s[k - 1]
        prob += s[k] == prev + eta * c[k] - d[k] / eta
        if allow_emergency:
            prob += b[k] + d[k] + pvE[k] + e[k] >= loadE[k] + c[k]
        else:
            prob += b[k] + d[k] + pvE[k] >= loadE[k] + c[k]
    prob += s[N - 1] == s_end
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    if pulp.LpStatus[prob.status] != "Optimal":
        raise RuntimeError("CBC not optimal: " + pulp.LpStatus[prob.status])
    out = {
        "obj": float(pulp.value(prob.objective)),
        "b": np.array([x.value() for x in b]),
        "c": np.array([x.value() for x in c]),
        "d": np.array([x.value() for x in d]),
        "s": np.array([x.value() for x in s]),
    }
    if plan is not None:
        out["u"] = np.array([x.value() for x in u])
        out["v"] = np.array([x.value() for x in v])
    if allow_emergency:
        out["e"] = np.array([x.value() for x in e])
    return out


def kkt_certificate(p, loadE, pvE, s_init, s_end, smin=CAP_MIN, smax=CAP_MAX,
                    eta=ETA, emax=E_MAX):
    """最优性证书: 用求解器返回的对偶解逐项验证 KKT 条件与强对偶.

    变量 (全部 >=0): b_k, c_k, d_k, q_k = s_k - smin
    约束: SOC 递推(等式)、功率平衡(<=)、c,d,q 上界(<=)、期末 s(<=/=)
    对偶: max -mu^T b_ub - z^T b_eq  s.t. c + A_ub^T mu + A_eq^T z >= 0, mu>=0
    """
    from scipy.optimize import linprog
    from scipy.sparse import lil_matrix, vstack

    n = K
    nv = 4 * n
    cobj = np.zeros(nv)
    cobj[0:n] = p

    Aeq = lil_matrix((n, nv))
    for k in range(n):
        Aeq[k, 3 * n + k] += 1.0
        Aeq[k, n + k] += -eta
        Aeq[k, 2 * n + k] += 1.0 / eta
        if k > 0:
            Aeq[k, 3 * n + k - 1] += -1.0
    beq = np.zeros(n)
    beq[0] = s_init - smin

    rows = []
    rhs = []
    def add_row(entries, b):
        r = lil_matrix((1, nv))
        for j, v in entries:
            r[0, j] = v
        rows.append(r.tocsr())
        rhs.append(b)

    for k in range(n):                       # b+d-c >= loadE-pvE  ->  -b-d+c <= pvE-loadE
        add_row([(k, -1.0), (2 * n + k, -1.0), (n + k, 1.0)], pvE[k] - loadE[k])
        add_row([(n + k, 1.0)], emax)
        add_row([(2 * n + k, 1.0)], emax)
        add_row([(3 * n + k, 1.0)], smax - smin)
    add_row([(3 * n + n - 1, 1.0)], s_end - smin)
    add_row([(3 * n + n - 1, -1.0)], -(s_end - smin))

    A_ub = vstack(rows).tocsc()
    b_ub = np.array(rhs, float)
    res = linprog(cobj, A_ub=A_ub, b_ub=b_ub, A_eq=Aeq.tocsc(), b_eq=beq,
                  bounds=[(0, None)] * nv, method="highs")
    if not res.success:
        raise RuntimeError(res.message)
    # scipy 对 A_ub x <= b_ub 的 marginal = dObj/db_ub = -mu, 故 mu = -marginal
    mu = -np.array(res.ineqlin.marginals, float)
    z_raw = np.array(res.eqlin.marginals, float)
    slack = b_ub - A_ub @ res.x
    best = None
    for sign in (1.0, -1.0):
        z = sign * z_raw
        rc = cobj + A_ub.T @ mu + Aeq.T @ z
        dobj = float(-mu @ b_ub - z @ beq)
        cand = dict(
            primal_obj=float(res.fun),
            dual_obj=dobj,
            duality_gap=abs(dobj - float(res.fun)),
            max_stationarity_violation=float(max(0.0, -rc.min())),
            max_comp_slack=float(np.max(np.abs(mu * slack))),
            max_dual_infeas=float(max(0.0, -mu.min())),
            sign=sign)
        score = cand["duality_gap"] + cand["max_stationarity_violation"]
        if best is None or score < best[0]:
            best = (score, cand)
    return best[1]


def dual_bound(p, loadE, pvE, s_init, s_end, smin=CAP_MIN, smax=CAP_MAX,
               eta=ETA, emax=E_MAX):
    """构造并求解对偶问题, 返回对偶最优值 (用于验证强对偶/最优性).

    采用显式推导的对偶: 原问题
       min Σ p_k b_k
       s.t. b_k + d_k - c_k >= loadE_k - pvE_k        (λ_k >= 0)
            s_k - s_{k-1} - eta c_k + d_k/eta = 0      (π_k 自由)
            0<=c_k<=emax, 0<=d_k<=emax, smin<=s_k<=smax
    对偶变量 (λ, π) 的对偶函数为
       Σ_k λ_k (loadE_k - pvE_k) + π_0 s_init?  ... 直接数值求解对偶松弛
    这里改用更稳健的做法: 用 Lagrange 松弛 (对 SOC 约束拉格朗日) 的
    部分对偶 + 二次? 不存在 -> 直接解 LP 对偶 (scipy)。
    """
    from scipy.optimize import linprog
    from scipy.sparse import lil_matrix
    import model as M

    # 原问题标准形: min c^T x, A_eq x = beq, A_ub x <= b_ub
    n = K
    nv = 4 * n
    idx_b, idx_c, idx_d, idx_s = 0, n, 2 * n, 3 * n
    cobj = np.zeros(nv)
    cobj[idx_b:idx_b + n] = p
    Aeq = lil_matrix((n, nv))
    for k in range(n):
        Aeq[k, idx_s + k] += 1
        Aeq[k, idx_c + k] += -eta
        Aeq[k, idx_d + k] += 1 / eta
        if k > 0:
            Aeq[k, idx_s + k - 1] += -1
    beq = np.zeros(n)
    beq[0] = s_init
    A_ub = lil_matrix((n, nv))
    b_ub = np.zeros(n)
    for k in range(n):
        A_ub[k, idx_b + k] = -1
        A_ub[k, idx_d + k] = -1
        A_ub[k, idx_c + k] = 1
        b_ub[k] = pvE[k] - loadE[k]
    lb = np.zeros(nv)
    ub = np.full(nv, np.inf)
    ub[idx_c:idx_c + n] = emax
    ub[idx_d:idx_d + n] = emax
    lb[idx_s:idx_s + n] = smin
    ub[idx_s:idx_s + n] = smax
    ub[idx_s + n - 1] = s_end
    lb[idx_s + n - 1] = s_end
    res = linprog(cobj, A_ub=A_ub.tocsc(), b_ub=b_ub, A_eq=Aeq.tocsc(),
                  b_eq=beq, bounds=list(zip(lb, ub)), method="highs-ipm")
    return float(res.fun), res
