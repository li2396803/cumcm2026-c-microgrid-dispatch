"""v2 改进 3: 储能寿命 (循环深度) 建模.

(1) 线性吞吐成本模型 (用于优化):  目标 += c_deg * Σ (η c + d/η)
(2) DoD 相关寿命模型 (用于评价):  N(DoD) = N0 * (DoD/DoD_ref)^(-k)
    -> 每次循环的损耗费用 = C_cap * 可用容量 / N(DoD)
(3) 雨流计数 (rainflow) 统计循环深度分布与年寿命消耗
(4) 迭代线性化: 用雨流结果修正 c_deg 后重优化, 直至收敛
"""
from __future__ import annotations

import numpy as np

import model as M

CAP_USABLE = 9600.0          # kWh, 1200-10800
CAP_INSTALLED = 12000.0      # kWh
COST_PER_KWH = 1000.0        # 元/kWh 装机 (LFP 系统)
N0, DOD_REF, K_WOHLER = 6000.0, 0.9, 1.6


def rainflow(series):
    """ASTM E1049 三点法雨流计数. 返回 (循环幅值列表, 残余半循环列表)."""
    x = np.asarray(series, float)
    if len(x) < 3:
        return np.array([]), np.array([])
    turning = [x[0]]
    for i in range(1, len(x) - 1):
        d1, d2 = x[i] - x[i - 1], x[i + 1] - x[i]
        if d1 * d2 < 0:
            turning.append(x[i])
    turning.append(x[-1])
    stack, cyc = [], []
    for v in turning:
        stack.append(v)
        while len(stack) >= 3:
            a, b, c = stack[-3], stack[-2], stack[-1]
            if abs(b - a) <= abs(c - b):
                cyc.append(abs(b - a))
                del stack[-3:-1]
            else:
                break
    rem = [abs(stack[i + 1] - stack[i]) for i in range(len(stack) - 1)]
    return np.array(cyc), np.array(rem)


def cycle_cost(depth_range):
    """单次循环 (SOC 摆幅 depth_range kWh) 的寿命损耗费用 (元)."""
    dod = np.clip(np.asarray(depth_range, float) / CAP_USABLE, 1e-6, 1.0)
    life = N0 * (dod / DOD_REF) ** (-K_WOHLER)
    cap_cost = COST_PER_KWH * CAP_INSTALLED
    return cap_cost / life


def life_fraction(depth_range):
    dod = np.clip(np.asarray(depth_range, float) / CAP_USABLE, 1e-6, 1.0)
    return (dod / DOD_REF) ** K_WOHLER / N0


def analyse(soc_traj):
    """对一条 SOC 轨迹做雨流统计, 返回寿命消耗与等效成本."""
    cyc, rem = rainflow(soc_traj)
    res = {"n_full_cycles": int(len(cyc)),
           "throughput": float(np.sum(M.ETA * 0 + 0))}
    life = float(life_fraction(cyc).sum()) if len(cyc) else 0.0
    # 残余半循环按 0.5 次计
    life_half = float(0.5 * life_fraction(rem).sum()) if len(rem) else 0.0
    cost = float(cycle_cost(cyc).sum()) + float(0.5 * cycle_cost(rem).sum())
    res["life_fraction"] = life + life_half
    res["degradation_cost"] = cost
    res["cyc"] = cyc
    res["rem"] = rem
    return res


def throughput(soc_or_flows, c=None, d=None):
    if c is None:
        s = np.asarray(soc_or_flows, float)
        return float(np.abs(np.diff(s)).sum())
    return float(np.sum(M.ETA * np.asarray(c) + np.asarray(d) / M.ETA))


def linear_equiv_cost(c, d):
    """与线性模型等价的单位 SOC 变化成本 (用于迭代标定)."""
    thr = throughput(None, c, d)
    return thr
