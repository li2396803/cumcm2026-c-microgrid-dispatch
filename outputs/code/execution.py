"""实时执行规则 (两种) —— 用于问题 3/4 的对比分析.

规则 A (计划跟踪, plan-tracking):
    储能的充放电严格按最近一次优化给出的计划执行; 实际光伏与预报的偏差
    直接由紧急购电(缺口)或弃光(盈余)消化。

规则 B (实时平衡, real-time balancing):
    储能在其功率与储电量范围内实时吸收光伏偏差:
      缺口 -> 先减少计划充电, 再增加放电, 仍不足才紧急购电;
      盈余 -> 先减少计划放电, 再增加充电, 仍有余量则弃光。
"""
from __future__ import annotations

import numpy as np

from model import ETA, E_MAX, CAP_MIN, CAP_MAX


def run_interval_ruleA(b, c, d, p, loadE, pvE_act, s_prev,
                       emergency_mult=5.0):
    """规则 A: 储能严格按计划执行."""
    e = np.maximum(0.0, loadE + c - d - b - pvE_act)
    spill = np.maximum(0.0, b + d + pvE_act - loadE - c)
    s = s_prev + np.cumsum(ETA * c - d / ETA)
    return {"c": c.copy(), "d": d.copy(), "s": s, "e": e, "spill": spill,
            "emergency_cost": float(np.sum(emergency_mult * p * e)),
            "emergency_energy": float(e.sum()),
            "spill_energy": float(spill.sum())}


def run_interval_ruleB(b, c, d, p, loadE, pvE_act, s_prev,
                       emergency_mult=5.0, eta=ETA,
                       cmax=E_MAX, dmax=E_MAX,
                       smin=CAP_MIN, smax=CAP_MAX):
    """规则 B: 储能在物理约束内实时平衡.

    关键恒等式: 储能需要向母线净提供的电量
        vol_target = loadE - b - pv_act
    (与计划的 c,d 无关), 只需在功率与储电量约束下截断:
        vol_target >= 0  -> 放电 min(vol_target, 功率上限, 可用电量)
        vol_target <  0  -> 充电 min(-vol_target, 功率上限, 可用容量)
    截断后仍不足的部分即为紧急购电, 充电装不下的部分即为弃光。
    因此该规则下 "购电计划 b 完全决定了储能的运行"。
    """
    n = len(b)
    c_ex = np.array(c, float)
    d_ex = np.array(d, float)
    e = np.zeros(n)
    spill = np.zeros(n)
    s = np.empty(n)
    cur = s_prev
    for k in range(n):
        vol_target = loadE[k] - b[k] - pvE_act[k]
        if vol_target >= 0.0:                             # 需要放电
            dd = min(vol_target, dmax, max(0.0, (cur - smin) * eta))
            cc = 0.0
        else:                                             # 需要充电
            cc = min(-vol_target, cmax, max(0.0, (smax - cur) / eta))
            dd = 0.0
        c_ex[k], d_ex[k] = cc, dd
        cur = cur + eta * cc - dd / eta
        s[k] = cur
        covered = dd - cc
        if vol_target > covered + 1e-9:
            e[k] = vol_target - covered
        elif covered > vol_target + 1e-9:
            spill[k] = covered - vol_target
    return {"c": c_ex, "d": d_ex, "s": s, "e": e, "spill": spill,
            "emergency_cost": float(np.sum(emergency_mult * p * e)),
            "emergency_energy": float(e.sum()),
            "spill_energy": float(spill.sum())}
