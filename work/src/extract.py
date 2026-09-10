"""Extract all C题 attachments into a single npz + json cache (work/out/data.npz).

Conventions
-----------
* 附件1 : one representative day (2025-01-01) with 144 ten-minute records,
          columns = [电价, 小区负载, 光伏发电预测功率]; the row label of each
          record is the LEFT endpoint of the 10-min slot it describes.
* 附件2 : 2025-01-01..12-31, 144 records/day, 小区负载 (kW), 光伏实际 (kW).
* 附件3 : 2025-1-1..12-31 x {0:00,6:00,12:00,18:00} x 24 hourly PV forecasts (kW).
* 附件4 : 2025-01-01..12-31, 144 records/day, 实时电价 (元/kWh).
"""
import datetime as dt
import json
import os

import numpy as np
import openpyxl

BASE = "/Users/lizelin/Downloads/CUMCM2026Problems/C题/附件"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out")
os.makedirs(OUT, exist_ok=True)

K = 144          # ten-minute slots per day
DT = 1.0 / 6.0   # slot length in hours


def _rows(path, sheet=None):
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb[sheet] if sheet else wb.worksheets[0]
    data = [list(r) for r in ws.iter_rows(values_only=True)]
    wb.close()
    return data


def _num(x):
    return 0.0 if x is None else float(x)


def load_att1():
    data = _rows(os.path.join(BASE, "附件1.xlsx"))
    labels = [r[0] for r in data[1:]]
    arr = np.array([[_num(r[1]), _num(r[2]), _num(r[3])] for r in data[1:]])
    assert arr.shape == (K, 3), arr.shape
    return labels, arr   # price, load(kW), pv_forecast(kW)


def load_att2():
    load = _rows(os.path.join(BASE, "附件2.xlsx"), "小区负载")
    pv = _rows(os.path.join(BASE, "附件2.xlsx"), "光伏发电实际功率")
    dates = [r[0] for r in load[1:]]
    dates2 = [r[0] for r in pv[1:]]
    assert dates == dates2, "date mismatch between the two sheets of 附件2"
    loadM = np.array([[_num(v) for v in r[1:]] for r in load[1:]])
    pvM = np.array([[_num(v) for v in r[1:]] for r in pv[1:]])
    assert loadM.shape == (365, K) and pvM.shape == (365, K)
    return dates, loadM, pvM


def load_att3():
    data = _rows(os.path.join(BASE, "附件3.xlsx"))
    dates, times, fc = [], [], []
    cur = None
    labels = [str(r[1]) for r in data[1:]]
    for r in data[1:]:
        d = r[0]
        if d not in (None, ""):
            cur = str(d)
        dates.append(cur)
        times.append(str(r[1]))
        fc.append([_num(v) for v in r[2:26]])
    fc = np.array(fc)
    assert fc.shape == (1460, 24), fc.shape
    return dates, times, fc


def load_att4():
    data = _rows(os.path.join(BASE, "附件4.xlsx"))
    dates = [r[0] for r in data[1:]]
    price = np.array([[_num(v) for v in r[1:]] for r in data[1:]])
    assert price.shape == (365, K), price.shape
    return dates, price


def main():
    l1, a1 = load_att1()
    d2, load2, pv2 = load_att2()
    d3, t3, fc3 = load_att3()
    d4, p4 = load_att4()

    # ---- consistency checks -------------------------------------------------
    day0 = dt.datetime(2025, 1, 1)
    info = {}
    info["att1_vs_att2_load_maxabs"] = float(np.max(np.abs(a1[:, 1] - load2[0])))
    info["att1_vs_att2_pv_maxabs"] = float(np.max(np.abs(a1[:, 2] - pv2[0])))
    info["att1_vs_att4_price_maxabs"] = float(np.max(np.abs(a1[:, 0] - p4[0])))
    info["att1_price_min"] = float(a1[:, 0].min())
    info["att1_price_max"] = float(a1[:, 0].max())
    info["att2_load_min"] = float(load2.min())
    info["att2_load_max"] = float(load2.max())
    info["att2_pv_max"] = float(pv2.max())
    info["att4_price_min"] = float(p4.min())
    info["att4_price_max"] = float(p4.max())
    info["att4_price_mean"] = float(p4.mean())
    info["n_days_att2"] = int(load2.shape[0])
    info["att1_labels_first_last"] = [str(l1[0]), str(l1[-1])]
    info["att3_forecast_times"] = sorted(set(t3))
    # per-day aggregated PV potential
    info["pv_daily_energy_mean_kWh"] = float((pv2 * DT).sum(axis=1).mean())
    info["load_daily_energy_mean_kWh"] = float((load2 * DT).sum(axis=1).mean())
    info["pv_surplus_slots_per_day_mean"] = float(
        np.mean((pv2 > load2).sum(axis=1)))
    info["att3_nan"] = int(np.isnan(fc3).sum())
    # forecast error quick stats
    dset = set(d2)
    for t in ["0:00", "6:00", "12:00", "18:00"]:
        pass
    print(json.dumps(info, ensure_ascii=False, indent=1))

    np.savez_compressed(
        os.path.join(OUT, "data.npz"),
        labels1=np.array([str(x) for x in l1]),
        att1=a1, load=load2, pv=pv2, price_rt=p4, fc3=fc3,
        dates2=np.array([str(x)[:10] for x in d2]),
        dates3=np.array([str(x) for x in d3]),
        times3=np.array([str(x) for x in t3]),
        dates4=np.array([str(x)[:10] for x in d4]),
    )
    with open(os.path.join(OUT, "info.json"), "w") as f:
        json.dump(info, f, ensure_ascii=False, indent=1)
    print("saved ->", os.path.join(OUT, "data.npz"))


if __name__ == "__main__":
    main()
