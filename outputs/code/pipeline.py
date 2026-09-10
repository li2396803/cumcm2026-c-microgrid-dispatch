"""C题 主求解流程: 问题1/2/3/4 的建模与求解.

约定 (与附件 5 模板一致):
  一天 144 个 10 分钟时段, 第 k 个时段为 (t_{k-1}, t_k], t_k = k*10min;
  附件中标签为 t_k 的记录用于第 k 个时段。
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import (K, DT, ETA, E_MAX, CAP_MIN, CAP_MAX, S0_DEFAULT,
                   solve_day, simulate_execution, check_solution)
from execution import run_interval_ruleA, run_interval_ruleB

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "out")


def upsample_hourly(v24):
    """把 24 个整点值展开成 144 个 10 分钟值 (每小时重复 6 次)."""
    return np.repeat(np.asarray(v24, float), 6)


def load_data():
    d = np.load(os.path.join(OUT, "data.npz"), allow_pickle=True)
    return d


def plan_day(price_slot, loadE, pvE, s_init, s_end, solver="highs-ds"):
    return solve_day(price_slot, loadE, pvE, s_init=s_init, s_end=s_end,
                     solver=solver)


# ---------------------------------------------------------------- 问题 1 ----
def problem1(data):
    a1 = data["att1"]
    p, loadE, pvE = a1[:, 0], a1[:, 1] * DT, a1[:, 2] * DT
    sol = plan_day(p, loadE, pvE, S0_DEFAULT, S0_DEFAULT)
    sol["price"] = p
    sol["loadE"] = loadE
    sol["pvE"] = pvE
    return sol


# ---------------------------------------------------------------- 问题 2 ----
def _day_range(data):
    """2025-02-01 .. 2025-12-31 在附件2中的行号."""
    dates = [str(x) for x in data["dates2"]]
    idx = [i for i, x in enumerate(dates) if x >= "2025-02-01"]
    return idx


def problem2(data, price_mode="att1"):
    a1, load, pv, prt = data["att1"], data["load"], data["pv"], data["price_rt"]
    p_fix = a1[:, 0]
    idx = _day_range(data)
    res = []
    for i in idx:
        p = p_fix if price_mode == "att1" else prt[i]
        sol = plan_day(p, load[i] * DT, pv[i] * DT, S0_DEFAULT, S0_DEFAULT)
        sim = simulate_execution(sol["b"], sol["c"], sol["d"], p,
                                 load[i] * DT, pv[i] * DT)
        res.append({
            "row": int(i), "date": str(data["dates2"][i]),
            "obj": sol["obj"], "obj_full": sol["obj"] + sim["emergency_cost"],
            "emergency_energy": sim["emergency_energy"],
            "emergency_cost": sim["emergency_cost"],
            "b": sol["b"], "c": sol["c"], "d": sol["d"], "s": sol["s"],
            "s0": sol["s0"],
        })
    return res


# ---------------------------------------------------------------- 问题 3 ----
REV_SLOTS = [0, 36, 72, 108]      # 0:00, 6:00, 12:00, 18:00 对应的时段起点


def _forecast_slots(data, day_idx, stage, shift_slots=0, n_slots=K):
    """取第 stage 个预报时刻发布的预报, 展开为 144 个 10 分钟值.

    stage: 0->0:00, 1->6:00, 2->12:00, 3->18:00
    day_idx: 附件2/附件4 中的日序号 (0-based)
    返回长度 144 的数组 (单位 kW), 超出当日部分为 0.
    """
    row = data["fc3"][day_idx * 4 + stage]
    start_h = [0, 6, 12, 18][stage]
    out = np.zeros(K)
    for k in range(1, 25):                      # 预报第 k 小时 = [start+k-1, start+k)
        h0 = start_h + k - 1
        if h0 >= 24:
            break
        a = max(0, h0 * 6 + shift_slots)
        b = min(K, (h0 + 1) * 6 + shift_slots)
        if b > a:
            out[a:b] = row[k - 1]
    return out


def forecast_energy(data, day_idx, stage, shift_slots=0):
    """预报展开为 144 个时段的电量 (kWh)."""
    return _forecast_slots(data, day_idx, stage, shift_slots) * DT


def problem3(data, price_mode="att1", rev_slots=REV_SLOTS,
             forecast_stages=(0, 1, 2, 3), s0=S0_DEFAULT, s_end=S0_DEFAULT,
             execution="A", hedge=0.0, fc_func=None, fc_shift=0,
             verbose=False):
    """滚动预报 + 偏差结算 + 紧急购电 的逐日模拟.

    rev_slots 给出了允许调整的时段起点 (第一个必须是 0 = 日前计划),
    forecast_stages 给出各调整时刻可用的预报号 (0,1,2,3 对应 0/6/12/18 时).
    execution: "A" 计划跟踪 / "B" 实时平衡
    hedge: 计划阶段对光伏预报的保守修正系数 (0 = 不做保守修正,
           0.1 表示按 P̂(1+hedge) 规划, 即预留 10% 光伏裕量)
    fc_func: 可选, fc_func(day_row, stage_index) -> 144 维预报电量 (kWh),
             用于合成预报/额外预报时刻的情形
    """
    a1, load, pv, prt = data["att1"], data["load"], data["pv"], data["price_rt"]
    p_fix = a1[:, 0]
    idx = _day_range(data)
    out = []
    for i in idx:
        p = p_fix if price_mode == "att1" else prt[i]
        loadE = load[i] * DT
        pv_act = pv[i] * DT
        # --- 阶段 1: 日前计划 (0:00 预报) ---
        if fc_func is None:
            f0 = forecast_energy(data, i, forecast_stages[0], fc_shift)
        else:
            f0 = fc_func(i, 0)
        plan = plan_day(p, loadE, f0 * (1.0 - hedge), s0, s_end)
        bp, cp, dp, sp = plan["b"], plan["c"], plan["d"], plan["s"]
        cost_plan = float(np.sum(p * bp))
        # --- 滚动调整 ---
        b_exec = bp.copy()
        c_exec = cp.copy()
        d_exec = dp.copy()
        s_exec = sp.copy()
        emerg = np.zeros(K)
        spill = np.zeros(K)
        stage_cost = float(np.sum(p * bp))       # 含偏差结算的总购电费
        emerg_cost = 0.0
        for si, tau in enumerate(rev_slots):
            if si > 0:
                prev = rev_slots[si - 1]
                # 上一区间按执行计划运行, 用实际光伏结算缺口
                sim = _exec(b_exec[prev:tau], c_exec[prev:tau], d_exec[prev:tau],
                            p[prev:tau], loadE[prev:tau], pv_act[prev:tau],
                            s_exec[prev - 1] if prev > 0 else s0, execution)
                emerg[prev:tau] = sim["e"]
                spill[prev:tau] = sim["spill"]
                c_exec[prev:tau] = sim["c"]
                d_exec[prev:tau] = sim["d"]
                s_exec[prev:tau] = sim["s"]
                emerg_cost += sim["emergency_cost"]
            if tau == 0:
                continue
            # 用最新预报重优化剩余时段
            if fc_func is None:
                f = forecast_energy(data, i, forecast_stages[si], fc_shift)
            else:
                f = fc_func(i, si)
            sub = solve_day(p[tau:], loadE[tau:], f[tau:] * (1.0 - hedge),
                            s_init=s_exec[tau - 1], s_end=s_end,
                            plan=bp[tau:], solver="highs-ds")
            b_exec[tau:] = sub["b"]
            c_exec[tau:] = sub["c"]
            d_exec[tau:] = sub["d"]
            s_exec[tau:] = sub["s"]
        # 最后一段 (18:00-24:00)
        prev = rev_slots[-1]
        sim = _exec(b_exec[prev:], c_exec[prev:], d_exec[prev:],
                    p[prev:], loadE[prev:], pv_act[prev:],
                    s_exec[prev - 1] if prev > 0 else s0, execution)
        emerg[prev:] = sim["e"]
        spill[prev:] = sim["spill"]
        c_exec[prev:] = sim["c"]
        d_exec[prev:] = sim["d"]
        s_exec[prev:] = sim["s"]
        emerg_cost += sim["emergency_cost"]
        # 结算总费用: 已执行时段按计划价, 调整时段按 50%/150% 结算
        bp_dummy = bp
        u = np.maximum(0.0, bp - b_exec)
        v = np.maximum(0.0, b_exec - bp)
        settle = float(np.sum(p * np.minimum(bp, b_exec) + 0.5 * p * u
                              + 1.5 * p * v))
        total = settle + emerg_cost
        out.append({
            "row": int(i), "date": str(data["dates2"][i]),
            "plan_b": bp, "plan_c": cp, "plan_d": dp, "plan_s": sp,
            "b": b_exec, "c": c_exec, "d": d_exec, "s": s_exec,
            "emergency": emerg, "spill": spill,
            "cost_plan": cost_plan,           # 仅日前计划购电费
            "cost_settle": settle,            # 计划+调整结算费
            "cost_emergency": emerg_cost,
            "total": total,
            "dev_up": float(v.sum()), "dev_down": float(u.sum()),
            "spill_energy": float(spill.sum()),
            "soc_end": float(s_exec[-1]),
        })
    return out


def _exec(b, c, d, p, loadE, pvE_act, s_prev, execution):
    if execution == "A":
        return run_interval_ruleA(b, c, d, p, loadE, pvE_act, s_prev)
    return run_interval_ruleB(b, c, d, p, loadE, pvE_act, s_prev)


def main():
    data = load_data()
    summary = {}

    print("== 问题 1 ==")
    s1 = problem1(data)
    print("  最优购电费 %.2f 元, 购电量 %.2f kWh" % (s1["obj"], s1["b"].sum()))
    np.savez(os.path.join(OUT, "sol_p1.npz"),
             b=s1["b"], c=s1["c"], d=s1["d"], s=s1["s"], obj=s1["obj"])
    summary["p1"] = {"cost": s1["obj"], "energy": float(s1["b"].sum()),
                     "soc_min": float(s1["s"].min()),
                     "soc_max": float(s1["s"].max())}

    print("== 问题 2 ==")
    r2 = problem2(data, "att1")
    tot2 = sum(x["obj_full"] for x in r2)
    print("  全年(2/1-12/31) 购电费 %.2f 元; 紧急购电 %.4f kWh"
          % (tot2, sum(x["emergency_energy"] for x in r2)))
    np.savez(os.path.join(OUT, "sol_p2.npz"),
             dates=np.array([x["date"] for x in r2]),
             b=np.array([x["b"] for x in r2]),
             c=np.array([x["c"] for x in r2]),
             d=np.array([x["d"] for x in r2]),
             s=np.array([x["s"] for x in r2]),
             cost=np.array([x["obj_full"] for x in r2]))
    summary["p2"] = {"days": len(r2), "total_cost": tot2,
                     "emergency_total": float(sum(x["emergency_energy"] for x in r2))}

    print("== 问题 3 ==")
    r3 = problem3(data, "att1", execution="A")
    tot3 = sum(x["total"] for x in r3)
    print("  全年总费用 %.2f 元 (计划 %.2f + 紧急 %.2f)"
          % (tot3, sum(x["cost_settle"] for x in r3),
             sum(x["cost_emergency"] for x in r3)))
    r3b = problem3(data, "att1", execution="B")
    tot3b = sum(x["total"] for x in r3b)
    print("  [规则B 实时平衡] 全年总费用 %.2f 元" % tot3b)
    np.savez(os.path.join(OUT, "sol_p3.npz"),
             dates=np.array([x["date"] for x in r3]),
             plan_b=np.array([x["plan_b"] for x in r3]),
             b=np.array([x["b"] for x in r3]),
             c=np.array([x["c"] for x in r3]),
             d=np.array([x["d"] for x in r3]),
             s=np.array([x["s"] for x in r3]),
             emergency=np.array([x["emergency"] for x in r3]),
             cost=np.array([x["total"] for x in r3]))
    summary["p3"] = {"days": len(r3), "total": tot3,
                     "settle": float(sum(x["cost_settle"] for x in r3)),
                     "emergency": float(sum(x["cost_emergency"] for x in r3)),
                     "energy": float(sum(x["emergency"].sum() for x in r3)),
                     "total_B": tot3b,
                     "emergency_B": float(sum(x["cost_emergency"] for x in r3b)),
                     "energy_B": float(sum(x["emergency"].sum() for x in r3b)),
                     "spill": float(sum(x["spill_energy"] for x in r3)),
                     "dev_up": float(sum(x["dev_up"] for x in r3)),
                     "dev_down": float(sum(x["dev_down"] for x in r3))}

    print("== 问题 4 (波动电价) ==")
    r42 = problem2(data, "att4")
    tot42 = sum(x["obj_full"] for x in r42)
    r43 = problem3(data, "att4", execution="A")
    tot43 = sum(x["total"] for x in r43)
    r43b = problem3(data, "att4", execution="B")
    tot43b = sum(x["total"] for x in r43b)
    print("  p4-2 %.2f 元, p4-3 %.2f 元" % (tot42, tot43))
    print("  p4-3 [规则B] %.2f 元" % tot43b)
    np.savez(os.path.join(OUT, "sol_p42.npz"),
             dates=np.array([x["date"] for x in r42]),
             b=np.array([x["b"] for x in r42]),
             c=np.array([x["c"] for x in r42]),
             d=np.array([x["d"] for x in r42]),
             s=np.array([x["s"] for x in r42]),
             cost=np.array([x["obj_full"] for x in r42]))
    np.savez(os.path.join(OUT, "sol_p43.npz"),
             dates=np.array([x["date"] for x in r43]),
             plan_b=np.array([x["plan_b"] for x in r43]),
             b=np.array([x["b"] for x in r43]),
             c=np.array([x["c"] for x in r43]),
             d=np.array([x["d"] for x in r43]),
             s=np.array([x["s"] for x in r43]),
             emergency=np.array([x["emergency"] for x in r43]),
             cost=np.array([x["total"] for x in r43]))
    summary["p42"] = {"days": len(r42), "total_cost": tot42}
    summary["p43"] = {"days": len(r43), "total": tot43,
                      "settle": float(sum(x["cost_settle"] for x in r43)),
                      "emergency": float(sum(x["cost_emergency"] for x in r43)),
                      "energy": float(sum(x["emergency"].sum() for x in r43)),
                      "total_B": tot43b}

    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
