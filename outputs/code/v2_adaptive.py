"""v2 改进 4: 预报时刻的自适应获取 (把"是否使用新预报"内生化).

对每个候选预报时刻集合 S ⊆ {6:00,12:00,18:00} 求解同一场景树模型, 得到
    obj(S) = 仅使用 S 中各时刻预报时的最优期望费用,
    总费用(S) = obj(S) + |S| * c_fc
逐日取 argmin, 得到"按边际价值购买预报"的最优获取策略。
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model as M
import pipeline as P
import v2_tree as V

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "out")


def stale_scenarios(scen, avail):
    """按可用预报集合 avail (True/False, 长度 4) 生成"过期预报"场景集."""
    s = {"act": scen["act"], "donors": scen["donors"],
         "scale": scen["scale"], "fc": [None] * 4}
    s["fc"][0] = scen["fc"][0]
    for m in range(1, 4):
        if avail[m]:
            s["fc"][m] = scen["fc"][m]
        else:
            s["fc"][m] = s["fc"][m - 1]
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=7)
    ap.add_argument("--n-scen", dest="n_scen", type=int, default=64)
    ap.add_argument("--branches", default="4,3,2")
    ap.add_argument("--out", default="v2_adaptive.json")
    args = ap.parse_args()
    br = tuple(int(v) for v in args.branches.split(","))
    data = P.load_data()
    cache = V.make_cache(data)
    idx = P._day_range(data)[::args.step]
    p = data["att1"][:, 0]
    subsets = [s for s in itertools.product([0, 1], repeat=3)]
    res = {"days": [], "subsets": [list(s) for s in subsets],
           "obj": [], "actual": []}
    t0 = time.time()
    for cnt, i in enumerate(idx):
        loadE = data["load"][i] * M.DT
        scen = V.build_scenarios(data, i, window=75, n_scen=args.n_scen,
                                 cache=cache)
        objs, acts = [], []
        for sub in subsets:
            avail = (True,) + tuple(bool(x) for x in sub)
            sc = stale_scenarios(scen, avail)
            tree = V.build_tree(sc, br)
            r = V.solve_tree(p, loadE, tree, sc, objective="saa")
            objs.append(r["obj"])
            acts.append(r["mean_cost"])
        res["days"].append(str(data["dates2"][i]))
        res["obj"].append(objs)
        res["actual"].append(acts)
        if cnt % 5 == 0:
            print(f"{cnt+1}/{len(idx)} {res['days'][-1]} "
                  f"obj(all)={objs[-1]:,.0f} obj(none)={objs[0]:,.0f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    with open(os.path.join(OUT, args.out), "w") as f:
        json.dump(res, f, ensure_ascii=False)
    obj = np.array(res["obj"])                      # (n_day, 8)
    names = ["6:00", "12:00", "18:00"]
    print("\n=== 增益矩阵 (obj(none) - obj(S), 元/天) ===")
    base = obj[:, 0]
    for si, sub in enumerate(subsets):
        used = [names[k] for k in range(3) if sub[k]]
        print(f"  {'+'.join(used) if used else '不使用调整预报':28s} "
              f"平均节约 {np.mean(base - obj[:, si]):8.1f} 元/天")
    np.save(os.path.join(OUT, args.out.replace(".json", ".npy")), obj)
    print("saved", args.out)


if __name__ == "__main__":
    main()
