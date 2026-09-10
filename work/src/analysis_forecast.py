"""分析二: 预报时刻的价值 (是否需要引入其他时刻的预报).

(1) 用附件3 真实预报比较不同调整时刻组合的全年费用;
(2) 用标定过的预报误差模型生成任意时刻的合成预报, 评估加密预报时刻的边际效益。
"""
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


def error_model(data):
    """按 (预报时刻, 提前小时) 估计预报误差的均值/标准差 (kW)."""
    pv = data["pv"]
    fc = data["fc3"]
    mus, sds = {}, {}
    for stage, T in enumerate([0, 6, 12, 18]):
        rows = fc[stage::4, :]                    # (365, 24)
        mu = np.zeros(25)
        sd = np.zeros(25)
        cnt = np.zeros(25)
        for k in range(1, 25):
            h0 = T + k - 1
            if h0 >= 24:
                continue
            actual = pv[:, h0 * 6:(h0 + 1) * 6].mean(axis=1)
            err = rows[:, k - 1] - actual
            mu[k] = err.mean()
            sd[k] = err.std()
            cnt[k] = len(err)
        mus[T], sds[T] = mu, sd
    return mus, sds


def interp_params(mus, sds, tau, L):
    """由相邻预报时刻的误差参数线性插值出任意时刻 tau 的参数."""
    times = sorted(mus)
    if tau <= times[0]:
        return mus[times[0]][min(L, 24)], sds[times[0]][min(L, 24)]
    if tau >= times[-1]:
        return mus[times[-1]][min(L, 24)], sds[times[-1]][min(L, 24)]
    for a, b in zip(times[:-1], times[1:]):
        if a <= tau <= b:
            w = (tau - a) / (b - a)
            L = min(L, 24)
            m = (1 - w) * mus[a][L] + w * mus[b][L]
            s = (1 - w) * sds[a][L] + w * sds[b][L]
            return m, s
    raise ValueError(tau)


def make_hybrid_provider(data, rev_hours, real_stages, mus, sds,
                         seed=20260910):
    """混合预报: 已知时刻用附件3 真实预报, 其余时刻用标定模型生成合成预报."""
    pv = data["pv"]
    rng = np.random.default_rng(seed)
    # 以 (日, 小时, 提前量) 索引, 使不同调整时刻表共享同一随机数
    noise = rng.standard_normal((365, 24, 25))
    # 误差幅度按日内光伏"气候态"比例缩放, 避免夜间产生虚假光伏
    clim = np.array([pv[:, h * 6:(h + 1) * 6].mean() for h in range(24)])
    w = clim / clim.max()

    def provider(i, si):
        tau = rev_hours[si]
        if tau in real_stages:
            return P.forecast_energy(data, i, real_stages[tau])
        out = np.zeros(M.K)
        for k in range(1, 25):
            h0 = tau + k - 1
            if h0 >= 24:
                break
            L = min(k, 24)
            m, s = interp_params(mus, sds, tau, L)
            actual = pv[i, h0 * 6:(h0 + 1) * 6].mean()
            val = actual + (m + s * noise[i, h0, L]) * w[h0]
            out[h0 * 6:(h0 + 1) * 6] = max(0.0, val)
        return out * M.DT

    return provider


def main():
    data = P.load_data()
    mus, sds = error_model(data)
    res = {"error_model": {str(T): {"lead": list(range(1, 25)),
                                    "mean_kW": mus[T][1:].round(2).tolist(),
                                    "sd_kW": sds[T][1:].round(2).tolist()}
                           for T in [0, 6, 12, 18]},
           "real_forecast_schedules": {}, "synthetic_schedules": {},
           "hedge_optimal_real": {}}
    print("预报误差模型 (按预报时刻, 提前 1-12h):")
    for T in [0, 6, 12, 18]:
        print(f"  {T:>2}:00 mean", np.round(mus[T][1:13], 1))
        print(f"  {T:>2}:00 sd  ", np.round(sds[T][1:13], 1))

    # ---------- (1) 真实预报下的调整时刻价值 ----------
    schedules_real = {
        "0:00 only":        ([0], (0,)),
        "0:00+12:00":       ([0, 72], (0, 2)),
        "0:00+6:00+18:00":  ([0, 36, 108], (0, 1, 3)),
        "0:00+6:00+12:00+18:00": ([0, 36, 72, 108], (0, 1, 2, 3)),
    }
    for name, (slots, stages) in schedules_real.items():
        r = P.problem3(data, "att1", rev_slots=slots, forecast_stages=stages)
        tot = sum(x["total"] for x in r)
        em = sum(x["cost_emergency"] for x in r)
        res["real_forecast_schedules"][name] = {
            "total": tot, "emergency": em,
            "settle": sum(x["cost_settle"] for x in r)}
        print(f"[真实预报] {name:24s} 总费用 {tot:,.0f} 元 (紧急 {em:,.0f})",
              flush=True)

    # ---------- (2) 在真实 4 次预报基础上加密预报时刻 ----------
    real_stages = {0: 0, 6: 1, 12: 2, 18: 3}
    scheds = {
        "0:00,6:00,12:00,18:00 (真实)": [0, 6, 12, 18],
        "加 9:00,15:00": [0, 6, 9, 12, 15, 18],
        "加 3:00,9:00,15:00,21:00": [0, 3, 6, 9, 12, 15, 18, 21],
        "每 2 小时": list(range(0, 24, 2)),
    }
    base = None
    for name, hours in scheds.items():
        prov = make_hybrid_provider(data, hours, real_stages, mus, sds)
        slots = [h * 6 for h in hours]
        r = P.problem3(data, "att1", rev_slots=slots,
                       fc_func=prov, execution="A")
        tot = sum(x["total"] for x in r)
        em = sum(x["cost_emergency"] for x in r)
        if base is None:
            base = tot
        res["synthetic_schedules"][name] = {
            "n_times": len(hours), "total": tot, "emergency": em,
            "settle": sum(x["cost_settle"] for x in r),
            "delta_vs_base": tot - base}
        print(f"[加密预报] {name:32s} 总费用 {tot:,.0f} 元 (紧急 {em:,.0f}, "
              f"Δ {tot-base:+,.0f})", flush=True)

    with open(os.path.join(OUT, "analysis_forecast.json"), "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print("saved analysis_forecast.json")


if __name__ == "__main__":
    main()
