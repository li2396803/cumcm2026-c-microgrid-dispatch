"""微网-外网电力调控核心模型 (C题).

统一符号
========
K   = 144        一天 10 分钟时段数
DT  = 1/6 h      时段长度
L_k              时段 k 的小区负载功率 (kW)
P_k              时段 k 的光伏功率 (kW)
loadE_k = L_k*DT 时段 k 的负载电量 (kWh)
pvE_k   = P_k*DT 时段 k 的光伏电量 (kWh)
b_k              时段 k 的计划/执行购电量 (kWh)
c_k              时段 k 储能充电量 (交流侧, kWh)
d_k              时段 k 储能放电量 (供给负荷侧, kWh)
s_k              时段 k 结束时的储电量 (kWh)
eta = 0.9        单向充/放电效率
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix

K = 144
DT = 1.0 / 6.0

CAP_MAX = 10800.0      # 运行上限 (kWh)
CAP_MIN = 1200.0       # 运行下限 (kWh)
P_MAX = 5000.0         # 最大充放电功率 (kW)
E_MAX = P_MAX * DT     # 每时段最大充/放电量 (kWh)
ETA = 0.9
S0_DEFAULT = 6000.0    # 2025-01-01 0:00 储电量


def solve_day(p, loadE, pvE, s_init=S0_DEFAULT, s_end=S0_DEFAULT,
              plan=None, fixed=None, allow_emergency=False,
              emergency_mult=5.0, eta=ETA, cmax=E_MAX, dmax=E_MAX,
              smin=CAP_MIN, smax=CAP_MAX, solver="highs", deg_cost=0.0):
    """单日线性规划.

    p        : (K,) 电价 元/kWh
    loadE    : (K,) 负荷电量 kWh
    pvE      : (K,) 光伏电量 kWh
    plan     : (K,) 日前计划购电量; 不为 None 时启用 50%/150% 偏差结算
    fixed    : (K,) bool, True 表示该时段购电量锁定为 plan (已执行)
    allow_emergency : 是否允许紧急购电 (软约束, 5 倍电价)

    目标(元):
      无 plan   : sum p_k b_k
      有 plan   : sum [ p_k b_k + 0.5 p_k u_k + 0.5 p_k v_k ], b = plan - u + v
                  (等价于: 削减部分按 50% 结算, 增加部分按 150% 结算)
      紧急购电  : + 5 sum p_k e_k
    """
    p = np.asarray(p, float)
    loadE = np.asarray(loadE, float)
    pvE = np.asarray(pvE, float)
    n = len(p)          # 允许子区间(如 6:00 之后的 108 个时段)
    idx = {}
    nv = 0

    def add(name, size):
        nonlocal nv
        idx[name] = slice(nv, nv + size)
        nv += size

    add("b", n)          # 购电量
    add("c", n)          # 充电量
    add("d", n)          # 放电量
    add("s", n)          # 时段末储电量
    if plan is not None:
        add("u", n)      # 相对计划的削减量
        add("v", n)      # 相对计划的增加量
    if allow_emergency:
        add("e", n)      # 紧急购电量
    nvar = nv

    cobj = np.zeros(nvar)
    cobj[idx["b"]] = p
    if deg_cost:
        # 储能寿命成本: 与 SOC 变化量成正比 (元 per kWh SOC 变化)
        cobj[idx["c"]] += deg_cost * eta
        cobj[idx["d"]] += deg_cost / eta
    if plan is not None:
        cobj[idx["u"]] = 0.5 * p
        cobj[idx["v"]] = 1.5 * p
    if allow_emergency:
        cobj[idx["e"]] = emergency_mult * p

    n_row_eq = n + (n if plan is not None else 0)
    Aeq = lil_matrix((n_row_eq, nvar))
    beq = np.zeros(n_row_eq)
    for k in range(n):
        Aeq[k, idx["s"].start + k] += 1.0
        Aeq[k, idx["c"].start + k] += -eta
        Aeq[k, idx["d"].start + k] += 1.0 / eta
        if k > 0:
            Aeq[k, idx["s"].start + k - 1] += -1.0
        beq[k] = s_init if k == 0 else 0.0
    if plan is not None:
        for k in range(n):
            r = n + k
            Aeq[r, idx["b"].start + k] = 1.0
            Aeq[r, idx["u"].start + k] = 1.0
            Aeq[r, idx["v"].start + k] = -1.0
            beq[r] = plan[k]

    A_ub = lil_matrix((n, nvar))
    b_ub = np.zeros(n)
    for k in range(n):
        A_ub[k, idx["b"].start + k] = -1.0
        A_ub[k, idx["d"].start + k] = -1.0
        A_ub[k, idx["c"].start + k] = 1.0
        if allow_emergency:
            A_ub[k, idx["e"].start + k] = -1.0
        b_ub[k] = pvE[k] - loadE[k]

    lb = np.zeros(nvar)
    ub = np.full(nvar, np.inf)
    ub[idx["c"]] = cmax
    ub[idx["d"]] = dmax
    lb[idx["s"]] = smin
    ub[idx["s"]] = smax
    ub[idx["s"].start + n - 1] = s_end
    lb[idx["s"].start + n - 1] = s_end
    if plan is not None:
        ub[idx["u"]] = np.maximum(plan, 0.0)
        if fixed is not None:
            fx = np.asarray(fixed, bool)
            ub[idx["u"].start:idx["u"].stop] = np.where(fx, 0.0, ub[idx["u"]])
            ub[idx["v"].start:idx["v"].stop] = np.where(fx, 0.0, ub[idx["v"]])

    res = linprog(cobj, A_ub=A_ub.tocsc(), b_ub=b_ub,
                  A_eq=Aeq.tocsc(), b_eq=beq,
                  bounds=list(zip(lb, ub)), method=solver)
    if not res.success:
        raise RuntimeError(f"LP failed: {res.message}")

    x = res.x
    out = {
        "status": int(res.status),
        "obj": float(res.fun),
        "b": x[idx["b"]],
        "c": x[idx["c"]],
        "d": x[idx["d"]],
        "s": x[idx["s"]],
        "s0": s_init,
    }
    if plan is not None:
        out["u"] = x[idx["u"]]
        out["v"] = x[idx["v"]]
    if allow_emergency:
        out["e"] = x[idx["e"]]
    out["cost_energy"] = float(np.sum(p * out["b"]))
    out["cost_settle"] = (float(np.sum(0.5 * p * (out["u"] + out["v"])))
                          if plan is not None else 0.0)
    out["cost_emergency"] = (float(np.sum(emergency_mult * p * out["e"]))
                             if allow_emergency else 0.0)
    return out


def simulate_execution(b_exec, c_exec, d_exec, p, loadE, pvE_act,
                       emergency_mult=5.0):
    """按给定 (b,c,d) 执行, 用实际光伏结算: 缺口 -> 紧急购电, 盈余 -> 弃光."""
    supply = b_exec + d_exec + pvE_act
    demand = loadE + c_exec
    e = np.maximum(0.0, demand - supply)
    spill = np.maximum(0.0, supply - demand)
    return {
        "e": e,
        "spill": spill,
        "cost": float(np.sum(p * b_exec) + np.sum(emergency_mult * p * e)),
        "emergency_energy": float(e.sum()),
        "emergency_cost": float(np.sum(emergency_mult * p * e)),
        "spill_energy": float(spill.sum()),
    }


def check_solution(sol, loadE, pvE, s_init, s_end, plan=None,
                   eta=ETA, cmax=E_MAX, dmax=E_MAX,
                   smin=CAP_MIN, smax=CAP_MAX):
    """独立复核: 逐时段重算 SOC 与功率平衡, 返回各违规量."""
    b, c, d, s = sol["b"], sol["c"], sol["d"], sol["s"]
    v = {}
    cur = s_init
    srec = np.empty(K)
    for k in range(K):
        cur = cur + eta * c[k] - d[k] / eta
        srec[k] = cur
    v["soc_recur_err"] = float(np.max(np.abs(srec - s)))
    v["soc_below_min"] = float(max(0.0, smin - s.min()))
    v["soc_above_max"] = float(max(0.0, s.max() - smax))
    v["soc_end_err"] = float(abs(s[-1] - s_end))
    v["charge_over_power"] = float(max(0.0, c.max() - cmax))
    v["disch_over_power"] = float(max(0.0, d.max() - dmax))
    v["negatives"] = float(max(0.0, -min(b.min(), c.min(), d.min())))
    bal = b + d + pvE - loadE - c
    v["balance_violation"] = float(max(0.0, -bal.min()))
    if plan is not None:
        v["plan_link_err"] = float(np.max(np.abs(b - (plan - sol["u"] + sol["v"]))))
    return v
