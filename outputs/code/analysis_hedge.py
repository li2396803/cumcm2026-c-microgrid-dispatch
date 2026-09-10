"""分析一: 日前计划的保守程度 (hedge) 与费用关系 —— 寻找预报不确定下的最优策略."""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline as P

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "out")


def main():
    data = P.load_data()
    res = {"att1": {}, "att4": {}}
    grid = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]
    for mode in ["att1", "att4"]:
        for h in grid:
            for ex in ["A"]:
                r = P.problem3(data, mode, execution=ex, hedge=h)
                tot = sum(x["total"] for x in r)
                em = sum(x["cost_emergency"] for x in r)
                st = sum(x["cost_settle"] for x in r)
                res[mode][f"{ex}_{h:.2f}"] = {"total": tot, "settle": st,
                                              "emergency": em,
                                              "em_energy": float(
                                                  sum(x["emergency"].sum() for x in r))}
                print(f"{mode} rule{ex} hedge={h:.2f} total={tot:,.0f} "
                      f"settle={st:,.0f} emerg={em:,.0f}", flush=True)
    with open(os.path.join(OUT, "analysis_hedge.json"), "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
