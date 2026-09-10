"""按附件 5 模板写出 result1/2/3/4-2/4-3.xlsx 结果文件.

行标签约定: 第 i 行(时间段) 使用附件中标签为 t_i 的记录, 即
    行标签 = t_{i-1}-t_i (0:00-0:10, 0:10-0:20, ..., 23:50-0:00+1)
与模板相比首行标签由 "0:10-0:20" 更正为 "0:00-0:10"(模板标签整体后移一格)。
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model as M
import pipeline as P

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "out")
DEST = os.path.join(os.path.dirname(ROOT), "outputs", "result_files")
os.makedirs(DEST, exist_ok=True)


def slot_labels():
    lab = []
    for k in range(144):
        a = k * 10
        b = (k + 1) * 10
        fa = f"{a//60}:{a%60:02d}"
        if a == 0:
            fa = "0:00"
        fb = "24:00" if b == 1440 else f"{b//60}:{b%60:02d}"
        lab.append(f"{fa}-{fb}")
    return lab


def block_rows():
    return [("0:00-4:00", 0, 24), ("4:00-8:00", 24, 48), ("8:00-12:00", 48, 72),
            ("12:00-16:00", 72, 96), ("16:00-20:00", 96, 120),
            ("20:00-24:00", 120, 144)]


def main():
    import openpyxl
    data = P.load_data()
    dates = [str(x) for x in data["dates2"]]
    idx = P._day_range(data)
    labels = slot_labels()
    blocks = block_rows()
    s1 = np.load(os.path.join(OUT, "sol_p1.npz"))
    s2 = np.load(os.path.join(OUT, "sol_p2.npz"), allow_pickle=True)
    s3 = np.load(os.path.join(OUT, "sol_p3.npz"), allow_pickle=True)
    s42 = np.load(os.path.join(OUT, "sol_p42.npz"), allow_pickle=True)
    s43 = np.load(os.path.join(OUT, "sol_p43.npz"), allow_pickle=True)
    p_fix = data["att1"][:, 0]

    # ---------------- result1 ----------------
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "计划购电量"
    ws.append(["时间段", "购电量"])
    for k in range(144):
        ws.append([labels[k], round(float(s1["b"][k]), 4)])
    ws2 = wb.create_sheet("充放电量")
    ws2.append(["时间段", "充电量", "放电量", "时刻", "储电量"])
    for name, a, b in blocks:
        ws2.append([name, round(float(s1["c"][a:b].sum()), 4),
                    round(float(s1["d"][a:b].sum()), 4), None, None])
    ws2.cell(row=2, column=4, value="0:00")
    ws2.cell(row=2, column=5, value=6000.0)
    ws2.cell(row=3, column=4, value="24:00")
    ws2.cell(row=3, column=5, value=round(float(s1["s"][-1]), 4))
    wb.save(os.path.join(DEST, "result1.xlsx"))

    def write_day_sheet(ws, B, C, D, S, plan_B=None, cost=None, plan_cost=None,
                        date_list=None):
        ws.append(["日期\\时间"] + labels + ["全天购电量", "全天购电费"])
        for j, i in enumerate(date_list):
            row = [str(dates[i])] + [round(float(v), 4) for v in B[j]]
            row += [round(float(B[j].sum()), 4),
                    round(float(cost[j]), 4)]
            ws.append(row)
        if plan_B is not None:
            ws.append(["计划购电量合计"] +
                      [round(float(plan_B[:, k].sum()), 4) for k in range(144)] +
                      [round(float(plan_B.sum()), 4),
                       round(float(np.sum(plan_cost)), 4)])

    def write_storage_sheet(ws, C, D, S, s0_list, date_list):
        ws.append(["日期", "时间段", "充电量", "放电量", "时刻", "储电量"])
        for j, i in enumerate(date_list):
            for bi, (name, a, b) in enumerate(blocks):
                ws.append([str(dates[i]) if bi == 0 else None, name,
                           round(float(C[j][a:b].sum()), 4),
                           round(float(D[j][a:b].sum()), 4),
                           "0:00" if bi == 0 else ("24:00" if bi == 1 else None),
                           round(float(s0_list[j]), 4) if bi == 0
                           else (round(float(S[j][-1]), 4) if bi == 1 else None)])

    def write_emergency(ws, em, date_list, prices):
        ws.append(["日期", "购电时间段", "购电量"])
        for j, i in enumerate(date_list):
            rows = [(labels[k], float(em[j][k]), float(prices[j][k]))
                    for k in range(144) if em[j][k] > 1e-6]
            if not rows:
                ws.append([str(dates[i]), "无", 0.0])
            for t, name in enumerate(rows):
                ws.append([str(dates[i]) if t == 0 else None, name[0],
                           round(name[1], 4)])

    # ---------------- result2 ----------------
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "计划购电量"
    c2 = s2["cost"]
    write_day_sheet(ws, s2["b"], s2["c"], s2["d"], s2["s"], cost=c2,
                    date_list=idx)
    ws = wb.create_sheet("充放电量")
    write_storage_sheet(ws, s2["c"], s2["d"], s2["s"], [6000.0] * len(idx), idx)
    ws = wb.create_sheet("紧急购电量")
    write_emergency(ws, np.zeros((len(idx), 144)), idx,
                    np.array([P.forecast_energy(data, i, 0) for i in idx]) * 0
                    + np.tile(p_fix, (len(idx), 1)))
    wb.save(os.path.join(DEST, "result2.xlsx"))

    # ---------------- result3 ----------------
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "计划购电量"
    plan_cost3 = np.array([float(np.sum(p_fix * s3["plan_b"][j]))
                           for j in range(len(idx))])
    write_day_sheet(ws, s3["plan_b"], None, None, None, cost=plan_cost3,
                    date_list=idx)
    ws = wb.create_sheet("调整购电量")
    write_day_sheet(ws, s3["b"], None, None, None, cost=s3["cost"],
                    date_list=idx)
    ws = wb.create_sheet("充放电量")
    write_storage_sheet(ws, s3["c"], s3["d"], s3["s"], [6000.0] * len(idx), idx)
    ws = wb.create_sheet("紧急购电量")
    write_emergency(ws, s3["emergency"], idx,
                    np.tile(p_fix, (len(idx), 1)))
    wb.save(os.path.join(DEST, "result3.xlsx"))

    # ---------------- result4-2 / result4-3 ----------------
    prt = np.array([data["price_rt"][i] for i in idx])
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "计划购电量"
    write_day_sheet(ws, s42["b"], None, None, None, cost=s42["cost"],
                    date_list=idx)
    ws = wb.create_sheet("充放电量")
    write_storage_sheet(ws, s42["c"], s42["d"], s42["s"],
                        [6000.0] * len(idx), idx)
    ws = wb.create_sheet("紧急购电量")
    write_emergency(ws, np.zeros((len(idx), 144)), idx, prt)
    wb.save(os.path.join(DEST, "result4-2.xlsx"))

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "计划购电量"
    plan_cost43 = np.array([float(np.sum(prt[j] * s43["plan_b"][j]))
                            for j in range(len(idx))])
    write_day_sheet(ws, s43["plan_b"], None, None, None, cost=plan_cost43,
                    date_list=idx)
    ws = wb.create_sheet("调整购电量")
    write_day_sheet(ws, s43["b"], None, None, None, cost=s43["cost"],
                    date_list=idx)
    ws = wb.create_sheet("充放电量")
    write_storage_sheet(ws, s43["c"], s43["d"], s43["s"],
                        [6000.0] * len(idx), idx)
    ws = wb.create_sheet("紧急购电量")
    write_emergency(ws, s43["emergency"], idx, prt)
    wb.save(os.path.join(DEST, "result4-3.xlsx"))
    print("wrote ->", DEST)
    print(sorted(os.listdir(DEST)))


if __name__ == "__main__":
    main()
