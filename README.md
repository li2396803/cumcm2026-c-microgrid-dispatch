# 2026 高教社杯全国大学生数学建模竞赛 C 题
## 微网与外部电网电力调控策略 —— 完整建模、求解与验证

本仓库包含 C 题从**数据读取 → 建模 → 求解 → 多方法检验 → 灵敏度分析 → 模型改进 → 假设复核**的全过程代码、
中间结果与最终文档，可一键复现论文中的全部数字。

---

## 1 问题与主要结果

| 问题 | 内容 | 结果 |
|---|---|---|
| 问题 1 | 代表日（附件 1）计划购电，硬约束、日闭环储电量 | 最优购电费 **35 126.95 元**，全天购电量 **59 482.70 kWh**，储能 0:00/24:00 均 6 000 kWh |
| 问题 2 | 固定电价（附件 1）+ 实际负载/光伏（附件 2），逐日 0:00 计划，缺口按 5 倍价紧急购电 | 全年（2025-02-01—12-31，334 天）**12 245 046.92 元**，紧急购电 **0 kWh**（命题 1：确定性最优解必无短缺） |
| 问题 3 | 0:00/6:00/12:00/18:00 光伏预报 + 偏差结算（少购 50%、多购 150%）+ 紧急购电 | 全年 **16 970 919.00 元**（计划+调整 12 800 700.29 + 紧急 4 170 218.71），紧急购电 976 388 kWh |
| 问题 4-2 | 波动电价（附件 4）下重算问题 2 | **12 830 486.39 元** |
| 问题 4-3 | 波动电价下重算问题 3 | **17 702 310.51 元** |

**核心建模结论**：完全信息下储能的唯一作用是时段套利（峰谷价比 3.2 远大于往返效率门槛 1.235）；
预报不确定下，紧急购电成为主要风险来源（问题 3 比问题 2 贵 38.6%），因此模型改进的重点是
**如何在不确定预报下做出可执行的购电计划**。

---

## 2 模型改进（v2）与效果

在满足题面要求的基线模型之上，实现了 5 项改进（详见 `docs/` 中的 v2 文档 §10–§15）：

| 策略 | 全年总费用（元，问题 3 口径） | 相对基线 | 紧急购电（kWh） | CVaR₉₀（元/日） | 最差单日（元） |
|---|---|---|---|---|---|
| 基线：确定性计划 + 计划跟踪执行 | 16 970 919 | — | 976 388 | 71 026 | 75 378 |
| v2-A：多阶段随机规划（SAA 场景树，96 情景） | 14 565 626 | −14.17% | 304 109 | 64 331 | 68 604 |
| v2-B：均值-CVaR（λ=0.5, α=0.9） | 14 574 754 | −14.12% | 229 559 | 64 364 | 68 239 |
| v2-C：有限模糊集鲁棒（最坏场景） | 14 668 064 | −13.57% | 211 082 | 64 686 | 68 553 |
| **v2-D：SAA 计划 + MPC 实时闭环** | **13 316 574** | **−21.53%** | **8 433** | **60 485** | **64 400** |
| 参考：完全信息下界（问题 2） | 12 245 047 | −27.84% | 0 | 56 823 | 59 971 |

v2-D 把基线与完全信息下界之间差距的 **77.3%** 补齐；波动电价下 v2-D 为 **13 901 097 元（−21.47%）**，
紧急购电 7 831 kWh。

其他改进要点：

- **储能寿命（EFC 口径）**：最优磨损价 0.104 元/kWh（LCOS 折算），问题 2 全年"能量+寿命"费再降 40 546 元，
  等效寿命 9.4 → 10.1 年；在含预报不确定性的情境下储能的对冲价值约 0.30 元/kWh，**不应**用寿命成本抑制储能使用。
- **预报时刻自适应获取**：6:00 / 12:00 / 18:00 三次预报更新的边际价值分别为 **684 / 529 / 0 元/日**；
  先做预报偏差校正可在此基础上再省 374 元/日。
- **假设逐条复核**（v2 文档 §16）：15 条关键假设均有数值复核；发现并修正 1 处建模缺陷
  （MPC 段内实时平衡会破坏日闭环储电量，最大偏差 489 kWh，已通过"末段严格跟随计划"修正）。

---

## 3 目录结构

```
.
├── data/                       原始数据（题目 PDF、附件 1–5）
├── docs/                       说明文档（PDF + Markdown）
│   ├── C题_..._完整建模说明文档_v2.pdf      ← 主交付物（40 页，含假设复核）
│   ├── C题_..._建模与求解说明文档.pdf        ← v1 基线完整文档（24 页）
│   └── C题_..._v2改进版说明文档.pdf         ← v2 改进专项（15 页）
├── outputs/
│   ├── result_files/           结果文件（附件 5 模板）
│   │   ├── result1.xlsx  result2.xlsx  result3.xlsx  result4-2.xlsx  result4-3.xlsx
│   │   └── result2_v2.xlsx  result3_v2.xlsx  result4-3_v2.xlsx     ← v2 改进版
│   ├── figures/                全部图表
│   ├── tables/                 结果表格（Markdown）与全部中间结果（JSON）
│   ├── code/                   求解、检验、分析、作图脚本
│   └── v2/                     v2 全年求解结果（.npz）
└── work/
    ├── src/                    全部源码（与 outputs/code 同源）
    └── out/                    中间结果（数据缓存、解文件、检验报告等）
```

---

## 4 复现步骤

环境：Python 3.12 + NumPy / SciPy(HiGHS) / OpenPyXL / Matplotlib；交叉验证另需 PuLP(CBC)。

```bash
cd work
PY=python3                                  # 需 Python 3.10+，含 numpy/scipy/openpyxl/matplotlib

# ---- 基线：四个子问题 ----
$PY src/extract.py          # 读取附件 → out/data.npz
$PY src/pipeline.py         # 问题 1/2/3/4 → out/sol_p*.npz
$PY src/verify_all.py       # 多方法检验（KKT、DP、逐时段复核）
$PY src/export_cbc_cases.py
$PY -m venv /tmp/venvcumcm && /tmp/venvcumcm/bin/pip install -q pulp numpy   # 交叉验证环境
/tmp/venvcumcm/bin/python src/cbc_check.py                                # CBC 独立求解器验证
$PY src/analysis_hedge.py   # 保守裕量扫描
$PY src/analysis_forecast.py# 预报时刻价值
$PY src/analysis_sensitivity.py
$PY src/write_results.py    # 输出 result1/2/3/4-2/4-3.xlsx
$PY src/make_figures.py && $PY src/dump_tables.py

# ---- v2：五项改进 ----
$PY src/v2_year.py --mode tree --objective saa  --out v2_tree_saa.npz
$PY src/v2_year.py --mode tree --objective cvar --lam 0.5 --alpha 0.9 --out v2_tree_cvar.npz
$PY src/v2_year.py --mode tree --objective robust --eps-mean 0.15 --out v2_tree_robust.npz
$PY src/v2_year.py --mode mpc  --objective saa --n-scen 96 --out v2_mpc_b.npz
$PY src/v2_year.py --mode mpc  --objective saa --price att4 --out v2_mpc_att4.npz
$PY src/v2_adaptive.py      # 预报时刻自适应获取
$PY src/v2_robust_tests.py  # 结算口径/联络线容量/时标稳健性
$PY src/v2_degrad.py && $PY src/v2_degrad_fp.py
$PY src/v2_write_results.py && $PY src/v2_figures.py && $PY src/v2_tables_final.py
```

全部计算确定性（合成预报实验固定随机种子），重跑后解文件逐字节一致。

---

## 5 检验证据（摘要）

| 检验 | 结果 |
|---|---|
| KKT + 强对偶最优性证书（问题 1） | 对偶间隙 0，平稳性违反 0，互补松弛 4.9×10⁻¹⁴ |
| 独立求解器 CBC 交叉验证 | 13 个算例目标值一致到 10⁻⁹ |
| 动态规划独立复算 | SOC 网格 15→5 kWh，间隙 0.21%→0.07%，收敛到 LP 最优 |
| 逐时段可行性复核 | 全年 48 096 个时段，全部约束违反量 ≤10⁻¹¹ |
| 与规则型策略对比 | 优化策略比固定时段规则节省 29.8% |
| 流水线复现 | 重跑后 5 个解文件 MD5 与首次完全一致 |

---

## 6 数据来源说明

`data/` 目录中的题目与附件来自竞赛主办方公开发布的题面材料，仅用于复现本仓库的计算结果，
详见 [`data/README.md`](data/README.md)。该目录可安全删除：`work/src/extract.py` 支持环境变量
`CUMCM_DATA_DIR` 指定附件目录，也可直接使用仓库内已缓存的 `work/out/data.npz`。

## 7 许可协议

本仓库作者原创的代码、数学模型与说明文档采用 **MIT License**（见 [`LICENSE`](LICENSE)），
可自由使用、修改、分发（保留版权声明即可）；许可覆盖范围见 [`NOTICE.md`](NOTICE.md)。

`data/` 目录中的竞赛题面与附件为**第三方材料**，版权归竞赛主办方所有，不在 MIT 许可范围内。

> 如需替换为其他协议：`Apache-2.0`（含专利授权，适合工程化复用）、`CC-BY-4.0`（适合文档/论文为主的项目）
> 都是常见选择；若希望代码与文档分别授权，可保留 MIT 覆盖代码、另为 `docs/` 增加 CC-BY-4.0 说明。

## 8 引用

若本仓库的方法或代码对你有帮助，欢迎引用：

```
Ada. (2026). CUMCM2026 C题：微网与外部电网电力调控策略 —— 完整建模、求解与验证.
GitHub: https://github.com/li2396803/cumcm2026-c-microgrid-dispatch (MIT License)
```
