"""
visualize.py
------------
生成三类图:
  1. 单被试信号总览 —— 原始/滤波信号、SCL、SCR 成分与检测到的峰值，标注实验分段
  2. 组水平配对图   —— 每个被试 baseline->stress 的连线，直观展示个体一致性
  3. 效应量汇总图   —— 各指标效应量与显著性
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 中文字体兜底，缺失时不报错只是显示为方框
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

C_RAW = "#B0B7C3"
C_CLEAN = "#1F4E79"
C_TONIC = "#E07B39"
C_PHASIC = "#2E8B57"


def plot_subject_overview(df: pd.DataFrame, scr: pd.DataFrame,
                          epochs_cfg: dict, subject: str,
                          out_path: str | Path, dpi: int = 150) -> None:
    """单被试信号总览图。"""
    fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
    t = df["time_sec"].to_numpy()

    # --- 面板 1: 原始 vs 清洗后 ---
    ax = axes[0]
    ax.plot(t, df["eda_raw"], color=C_RAW, lw=0.7, label="原始 raw")
    ax.plot(t, df["eda_clean"], color=C_CLEAN, lw=1.1, label="滤波后 filtered")
    if df["artifact"].any():
        ax.fill_between(t, ax.get_ylim()[0], ax.get_ylim()[1],
                        where=df["artifact"], color="red", alpha=0.12,
                        label="伪迹段 artifact")
    ax.set_ylabel("EDA (μS)")
    ax.set_title(f"被试 {subject} — EDA 信号处理总览", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8, ncol=3)

    # --- 面板 2: SCL (tonic) ---
    ax = axes[1]
    ax.plot(t, df["EDA_Tonic"], color=C_TONIC, lw=1.4)
    ax.set_ylabel("SCL (μS)")
    ax.set_title("紧张性成分 SCL — 整体唤醒水平", fontsize=10)

    # --- 面板 3: SCR (phasic) + 峰值 ---
    ax = axes[2]
    ax.plot(t, df["EDA_Phasic"], color=C_PHASIC, lw=0.9)
    if len(scr):
        ax.scatter(scr["peak_sec"], np.interp(scr["peak_sec"], t, df["EDA_Phasic"]),
                   color="crimson", s=22, zorder=5,
                   label=f"SCR 事件 (n={len(scr)})")
        ax.legend(loc="upper right", fontsize=8)
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.set_ylabel("SCR (μS)")
    ax.set_xlabel("时间 (秒)")
    ax.set_title("相位性成分 SCR — 事件性交感反应", fontsize=10)

    # 分段阴影
    colors = {"baseline": "#4A90D9", "stress": "#D9534F"}
    for a in axes:
        for label, w in epochs_cfg.items():
            a.axvspan(w["start"], w["end"], color=colors.get(label, "gray"), alpha=0.07)
    for label, w in epochs_cfg.items():
        axes[0].text((w["start"] + w["end"]) / 2, axes[0].get_ylim()[1],
                     label, ha="center", va="bottom", fontsize=9,
                     color=colors.get(label, "gray"), fontweight="bold")

    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
        a.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("已保存 %s", out_path)


def plot_paired_comparison(subject_metrics: pd.DataFrame, results: pd.DataFrame,
                           out_path: str | Path, dpi: int = 150) -> None:
    """组水平配对连线图: 每条线代表一个被试。"""
    metrics = [m for m in results["metric"] if m in subject_metrics.columns]
    if not metrics:
        return

    n = len(metrics)
    ncol = min(4, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 3.4 * nrow))
    axes = np.atleast_1d(axes).ravel()

    wide = subject_metrics.pivot(index="subject", columns="epoch")

    for i, m in enumerate(metrics):
        ax = axes[i]
        try:
            b = wide[(m, "baseline")].to_numpy()
            s = wide[(m, "stress")].to_numpy()
        except KeyError:
            ax.axis("off")
            continue

        for bi, si in zip(b, s):
            color = "#D9534F" if si > bi else "#4A90D9"
            ax.plot([0, 1], [bi, si], color=color, alpha=0.5, lw=1.2,
                    marker="o", ms=4)

        ax.plot([0, 1], [np.nanmean(b), np.nanmean(s)], color="black",
                lw=2.4, marker="s", ms=7, zorder=10, label="组均值")

        row = results[results["metric"] == m].iloc[0]
        p = row["p_value"]
        star = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "n.s."
        ax.set_title(f"{m}\n{row['test']}  p={p:.4f} {star}", fontsize=9)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["baseline", "stress"], fontsize=9)
        ax.set_xlim(-0.25, 1.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.2, axis="y")

    for j in range(len(metrics), len(axes)):
        axes[j].axis("off")

    fig.suptitle("baseline vs stress 配对比较 (每线一名被试)",
                 fontsize=13, fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("已保存 %s", out_path)


def plot_effect_sizes(results: pd.DataFrame, out_path: str | Path,
                      dpi: int = 150) -> None:
    """效应量水平条形图。"""
    res = results.dropna(subset=["effect_size"]).copy()
    if res.empty:
        return
    res = res.sort_values("effect_size")

    fig, ax = plt.subplots(figsize=(8, 0.55 * len(res) + 2))
    colors = ["#D9534F" if s else "#B0B7C3" for s in res["significant_uncorrected"]]
    ax.barh(res["metric"], res["effect_size"], color=colors, height=0.6)

    for ref, ls in [(0.2, ":"), (0.5, "--"), (0.8, "-.")]:
        ax.axvline(ref, color="gray", ls=ls, lw=0.8, alpha=0.6)
        ax.axvline(-ref, color="gray", ls=ls, lw=0.8, alpha=0.6)
    ax.axvline(0, color="black", lw=1)

    ax.set_xlabel("效应量 (Cohen's dz / r)")
    ax.set_title("各指标效应量 (红色 = 未校正 p<0.05)\n参考线: 0.2 小 / 0.5 中 / 0.8 大",
                 fontsize=11, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.2, axis="x")

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("已保存 %s", out_path)
