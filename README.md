# EDA / TSST-C 皮电应激反应分析管线

从 BIOPAC `.acq` 原始记录到统计结论的完整 EDA (皮肤电活动) 分析管线，
用于 TSST-C 心理应激范式下 **基线 (baseline) vs 应激后 (stress)** 的配对比较。

---

## 分析流程

```
.acq 原始记录
    │
    ├─ 1. 读取与降采样        load_acq.py     通道定位、抗混叠降采样
    ├─ 2. 预处理              preprocess.py   低通滤波、运动伪迹检测与插值
    ├─ 3. SCL/SCR 分解        decompose.py    cvxEDA 分离紧张性/相位性成分
    ├─ 4. SCR 事件检测        scr_detect.py   峰值检测、幅值与潜伏期提取
    ├─ 5. 分段指标提取        scr_detect.py   baseline / stress 各段指标
    └─ 6. 配对统计检验        stats.py        正态性检验 → t 检验 / Wilcoxon
                                              效应量 + Holm-Bonferroni 校正
```

---

## 核心概念

### SCL 与 SCR 的区别

| | SCL (紧张性 tonic) | SCR (相位性 phasic) |
|---|---|---|
| 时间尺度 | 分钟级缓慢漂移 | 1-3 秒上升，3-10 秒恢复 |
| 生理意义 | 整体唤醒水平 | 事件性交感激活 |
| 个体差异 | 极大，绝对值不可跨人比较 | 相对稳定 |
| 主要指标 | 均值、变化斜率 | 事件频次、幅值 |

SCR 又分两类：**ER-SCR** (event-related，刺激后 1-4 秒窗口内，可归因于特定事件)
与 **NS-SCR** (non-specific，无明确刺激对应)。TSST-C 是持续性应激任务而非离散刺激，
因此主要指标是 **NS-SCR 频次 (次/分钟)**，而非单次事件幅值。

### 分析指标

| 指标 | 含义 |
|---|---|
| `SCL_mean` | 平均皮电水平 (μS) |
| `SCL_slope` | 皮电水平线性趋势 (μS/min)，反映唤醒的持续上升 |
| `SCR_count` | 该段 SCR 事件总数 |
| `SCR_rate_per_min` | 每分钟 SCR 频次 — **段长不等时必须用这个** |
| `SCR_amplitude_mean` | 平均幅值 (μS) |
| `SCR_amplitude_sum` | 幅值总和，反映总体交感激活量 |
| `EDA_std` | 信号波动性 |

---

## 快速开始

### 安装

```bash
pip install -r requirements.txt
```

> `cvxopt` 是 cvxEDA 分解方法的必需依赖。若未安装，管线会自动回退到
> `highpass` 方法并给出警告 —— 但 highpass 精度差很多 (见下方"方法选择依据")。

### 第一步：确认通道名

不同 BIOPAC 配置下 EDA 通道的命名不同，先查看：

```bash
py run_analysis.py --inspect data/raw/S01.acq
```

输出示例：

```
 index          name  units  samples  sampling_rate
     0       EDA100C     uS   300000         1000.0
     1           ECG     mV   300000         1000.0
```

把 EDA 通道的 `name` 或 `index` 填入 `config.yaml`：

```yaml
acquisition:
  eda_channel_name: "EDA100C"     # 或用 eda_channel_index: 0
```

### 第二步：设置实验分段时间

在 `config.yaml` 中按记录起点的相对秒数填写：

```yaml
epochs:
  baseline: { start: 0,   end: 300 }
  stress:   { start: 300, end: 600 }
```

若每名被试的时间点不同，建立 `data/epochs.csv` 覆盖：

```csv
subject,epoch,start,end
S01,baseline,0,300
S01,stress,312,612
S02,baseline,0,300
S02,stress,298,598
```

### 第三步：运行

```bash
py run_analysis.py --data-dir data/raw
```

---

## 输出结构

```
results/
├── subject_metrics.csv        每名被试 × 每个分段的全部指标
├── statistics.csv             配对检验结果 (含效应量与校正 p 值)
├── all_scr_events.csv         所有 SCR 事件明细
├── per_subject/               逐被试 SCR 事件表
└── figures/
    ├── S01_overview.png              单被试信号总览
    ├── group_paired_comparison.png   组水平配对连线图
    └── effect_sizes.png              效应量汇总
```

---

## 方法选择依据

### 为什么用 cvxEDA 分解

在已知真值的合成信号上对比 (300 秒 @ 50 Hz)：

| 真实 SCR 数 | highpass 检出 | cvxEDA 检出 |
|---|---|---|
| 5  | 20 | 4 |
| 10 | 31 | 9 |
| 20 | 94 | 22 |

highpass 严重过检 (3-5 倍假阳性)，cvxEDA 基本还原真值。
cvxEDA 把 EDA 建模为 `tonic + phasic 卷积 + 噪声`，通过带约束的二次规划求解分离，
代价是速度较慢。

### SCR 幅值阈值如何确定

`amplitude_min` 直接决定检出事件数，文献常用范围 0.01-0.05 μS 跨度很大。
在合成信号上扫描标定 (`tools/calibrate_threshold.py --synthetic`)：

| amp_min | gt=5 | gt=10 | gt=20 | gt=40 | 平均相对误差 |
|---|---|---|---|---|---|
| 0.01 | 19.7 | 30.0 | 95.0 | 99.7 | 254.4% |
| 0.02 | 5.3 | 11.0 | 40.0 | 55.7 | 39.0% |
| 0.03 | 4.0 | 9.0 | 22.0 | 42.3 | 11.5% |
| **0.04** | **4.0** | **9.0** | **20.0** | **40.3** | **7.7%** |
| 0.05 | 4.0 | 9.0 | 19.0 | 39.0 | 9.4% |

取 `0.04` 为默认值。真实数据信噪比不同，建议用
`tools/calibrate_threshold.py --real data/raw` 复标定 ——
基线段的 NS-SCR 应落在生理合理区间 (静息约 1-5 次/分钟)。

### 运动伪迹检测经过两次修正

这个模块最初两版都是错的，过程记录在此以备查：

**v1 — 固定阈值 2 μS/s：失败。**
实测真实 SCR 的上升速率可达 6.5 μS/s (合成信号 p99.9 = 6.2)。
固定 2 μS/s 的结果是**每一个正常 SCR 波峰都被标记为伪迹**，
在总览图上表现为红色伪迹条精确地压在每个波峰上。

**v2 — median + k·MAD：仍失败。**
EDA 差分分布高度右偏 (绝大多数样本接近 0，少数 SCR 上升沿很大)，
median 与 MAD 都被压在接近 0 处，阈值过低。实测结果完全反了：

| 测试信号 | v2 误检率 |
|---|---|
| 干净信号 (10 个 SCR) | 4.47% |
| 密集 SCR (40 个) | 14.19% ← 假阳性 |
| 注入 3 次电极跳变 | 5.08% ← 真伪迹反而更低 |

**v3 — 高分位数缩放：通过。**
以 `p99.5(差分) × 3.0` 为阈值。p99.5 是该被试"正常信号动态范围"的稳健估计，
SCR 上升沿即使很陡也落在其附近，而电极移动造成的阶跃比它快一个数量级：

| 测试信号 | v3 检出率 | 预期 |
|---|---|---|
| 干净信号 | 0.00% | ~0 ✓ |
| 密集 SCR (40 个) | 0.00% | ~0 ✓ |
| 注入 3 次电极跳变 | 0.62% | >0 ✓ |

另设 `absolute_ceiling = 15 μS/s` 兜底：超过此速率生理上不可能是 SCR。

### 统计方法

1. 计算每名被试的差值 `stress − baseline`
2. Shapiro-Wilk 检验差值正态性
3. 正态 → 配对 t 检验；非正态 → Wilcoxon 符号秩检验
4. 效应量：t 检验用 **Cohen's dz** (配对设计专用，= 差值均值/差值标准差，
   注意与独立样本的 Cohen's d 不同)；Wilcoxon 用 **r = Z/√N**
5. 多重比较用 **Holm-Bonferroni** 校正 (比 Bonferroni 更不保守，
   同样严格控制族系误差率 FWER)

---

## 验证

不需要真实数据即可验证管线正确性：

```bash
py tests/test_pipeline.py
```

用 NeuroKit2 合成 12 名被试，stress 段人为设置更高的 SCL 与 SCR 频次，
管线应当检出显著差异。合成数据上的实际输出：

```
metric              baseline    stress    test       p_holm   effect_size
SCL_mean              1.639     6.749     paired_t   0.00000    55.87
SCR_rate_per_min      2.100    13.818     paired_t   0.00000     5.77
SCR_amplitude_sum    11.077    33.838     paired_t   0.00000    12.00
```

SCR 频次 2.1 → 13.8 次/分钟，落在生理合理区间
(静息 1-5 次/分钟，应激 10-20 次/分钟)。

> 注：合成数据的效应量极大是因为组间差异是人为设定的，
> 真实数据不会有这种量级。这里只验证管线能正确检出已知的差异方向。

---

## 项目结构

```
eda-tsst-analysis/
├── README.md
├── requirements.txt
├── config.yaml                 全部参数集中在此，不需要改代码
├── run_analysis.py             主入口
├── src/
│   ├── load_acq.py             .acq 读取、通道定位、降采样
│   ├── preprocess.py           低通滤波、伪迹检测与处理
│   ├── decompose.py            SCL/SCR 分解 (cvxEDA)
│   ├── scr_detect.py           SCR 事件检测、分段指标提取
│   ├── stats.py                配对统计检验、效应量、多重比较校正
│   └── visualize.py            三类图表生成
├── tools/
│   └── calibrate_threshold.py  SCR 幅值阈值标定
├── tests/
│   └── test_pipeline.py        合成数据端到端验证
└── data/
    ├── raw/                    放 .acq 文件 (不纳入版本控制)
    └── epochs.csv              可选：被试特定分段时间
```

---

## 环境依赖

Python 3.9+

| 包 | 用途 |
|---|---|
| `bioread` | 读取 BIOPAC `.acq`，无需 AcqKnowledge 软件 |
| `neurokit2` | SCL/SCR 分解与峰值检测 |
| `cvxopt` | cvxEDA 分解的求解器后端 |
| `scipy` | 滤波、重采样、统计检验 |
| `pandas` / `numpy` | 数据处理 |
| `matplotlib` | 图表 |
| `pyyaml` | 配置文件 |

---

## 已知限制

- 目前只支持 **两分段配对设计** (baseline vs stress)。多分段 (如加入 recovery)
  需要改为重复测量 ANOVA 或混合线性模型。
- 伪迹检测的分位数阈值在极端噪声记录 (伪迹占比 > 30%) 下会失效，
  因为伪迹本身抬高了 p99.5。这类记录管线会给出警告，建议人工检查后剔除。
- SCR 检测未做 ER-SCR 与 NS-SCR 的区分。若实验有精确的事件时间戳，
  可在 `scr_detect.py` 中按刺激后 1-4 秒窗口筛选。
