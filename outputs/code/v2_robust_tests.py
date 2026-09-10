"""v2 局限补强: 结算口径歧义 / 联络线容量 / 时标对齐 对 v2 策略的影响 (抽样检验)."""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model as M
import pipeline as P
import v2_tree as V
import v2_policy as POL

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "out")


def run_case(tag, step=9, n_scen=64, branches=(4, 3, 2), settle_alpha=0.5,
             b_max_kw=np.inf, shift=0, deg=0.0, window=75):
    """在抽样日上运行 SAA 树策略; shift 为预报/实测时间对齐偏移(时段数)."""
    data = P.load_data()
    cache = V.make_cache(data)
    idx = P._day_range(data)[::step]
    p = data["att1"][:, 0]
    tot, em, thr = 0.0, 0.0, 0.0
    t0 = time.time()
    for i in idx:
        loadE = data["load"][i] * M.DT
        scen = V.build_scenarios(data, i, window=window, n_scen=n_scen,
                                 cache=cache)
        if shift:
            # 对齐偏移: 目标日的预报整体平移 (模拟时标校正), 场景实际同步平移
            scen = {"act": np.roll(scen["act"], shift, axis=1),
                    "fc": [np.roll(f, shift, axis=1) for f in scen["fc"]],
                    "donors": scen["donors"], "scale": scen["scale"]}
        tree = V.build_tree(scen, branches)
        r = V.solve_tree(p, loadE, tree, scen, objective="saa",
                         settle_alpha=settle_alpha,
                         b_max=(b_max_kw * M.DT if np.isfinite(b_max_kw)
                                else np.inf), deg_cost=deg)
        ev = POL.evaluate_tree_policy(data, i, tree, r, deg_cost=deg)
        tot += ev["total"]
        em += ev["em_energy"]
        thr += float(np.sum(M.ETA * ev["c"] + ev["d"] / M.ETA))
    print(f"[{tag}] n={len(idx)} 总费用 {tot:,.0f} 元 紧急 {em:,.0f} kWh "
          f"吞吐 {thr:,.0f} kWh ({time.time()-t0:.0f}s)", flush=True)
    return {"tag": tag, "n": len(idx), "total": tot, "em": em, "thr": thr,
            "avg_total": tot / len(idx), "avg_em": em / len(idx)}


def main():
    res = []
    res.append(run_case("基准 SAA (口径B, 无容量限制)"))
    res.append(run_case("结算口径A", settle_alpha=1.5))
    res.append(run_case("联络线容量 8 MW", b_max_kw=8000.0))
    res.append(run_case("联络线容量 6 MW", b_max_kw=6000.0))
    res.append(run_case("时标校正 -30min", shift=-3))
    with open(os.path.join(OUT, "v2_robust_tests.json"), "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    base = res[0]["avg_total"]
    print("\n=== 抽样对比 (平均单日费用) ===")
    for r in res:
        print(f"  {r['tag']:32s} {r['avg_total']:9,.0f} 元/日 "
              f"({100*(r['avg_total']/base-1):+.2f}%)  紧急 {r['avg_em']:.0f} kWh/日")


if __name__ == "__main__":
    main()
