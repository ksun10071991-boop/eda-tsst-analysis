"""
src/stats.py
--------
baseline vs stress 配对统计检验。

方法选择逻辑:
  1. 计算每个被试的差值 (stress - baseline)
  2. Shapiro-Wilk 检验差值的正态性
  3. 正态 -> 配对样本 t 检验 (paired t-test)
     非正态 -> Wilcoxon 符号秩检验 (非参数替代)

效应量:
  Cohen's dz —— 配对设计专用，= 差值均值 / 差值标准差
                注意与独立样本的 Cohen's d 不同，不要混用
  r          —— Wilcoxon 对应的效应量，= Z / sqrt(N)

多重比较:
  同时检验多个指标会抬高假阳性率。这里提供 Holm-Bonferroni 校正
  (比 Bonferroni 更不保守，且同样严格控制族系误差率 FWER)。
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats as sps

logger = logging.getLogger(__name__)


def _holm_bonferroni(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni 校正，返回校正后 p 值。"""
    p = np.asarray(pvals, dtype=float)
    valid = ~np.isnan(p)
    out = np.full(len(p), np.nan)
    if valid.sum() == 0:
        return out.tolist()

    idx = np.where(valid)[0]
    order = idx[np.argsort(p[idx])]
    m = len(order)
    prev = 0.0
    for rank, i in enumerate(order):
        adj = (m - rank) * p[i]
        adj = max(adj, prev)      # 保证单调不减
        out[i] = min(adj, 1.0)
        prev = out[i]
    return out.tolist()


def paired_test(baseline: np.ndarray, stress: np.ndarray,
                metric: str, config: dict) -> dict:
    """对单个指标做配对检验。"""
    cfg = config["statistics"]

    b = np.asarray(baseline, dtype=float)
    s = np.asarray(stress, dtype=float)
    valid = ~(np.isnan(b) | np.isnan(s))
    b, s = b[valid], s[valid]
    n = len(b)

    result = {
        "metric": metric,
        "n": n,
        "baseline_mean": float(np.mean(b)) if n else np.nan,
        "baseline_sd": float(np.std(b, ddof=1)) if n > 1 else np.nan,
        "stress_mean": float(np.mean(s)) if n else np.nan,
        "stress_sd": float(np.std(s, ddof=1)) if n > 1 else np.nan,
        "mean_diff": float(np.mean(s - b)) if n else np.nan,
        "test": None, "statistic": np.nan, "p_value": np.nan,
        "effect_size": np.nan, "effect_size_type": None,
        "normality_p": np.nan,
    }

    if n < 3:
        logger.warning("指标 %s 有效样本仅 %d 例，跳过检验。", metric, n)
        result["test"] = "insufficient_n"
        return result

    diff = s - b
    if np.allclose(diff, 0):
        logger.warning("指标 %s 差值全为 0，跳过检验。", metric)
        result["test"] = "zero_variance"
        result["p_value"] = 1.0
        return result

    # 正态性检验
    try:
        _, norm_p = sps.shapiro(diff)
        result["normality_p"] = float(norm_p)
    except Exception:
        norm_p = 1.0

    if norm_p > cfg.get("normality_alpha", 0.05):
        stat, p = sps.ttest_rel(s, b)
        sd = np.std(diff, ddof=1)
        result.update({
            "test": "paired_t",
            "statistic": float(stat),
            "p_value": float(p),
            "effect_size": float(np.mean(diff) / sd) if sd > 0 else np.nan,
            "effect_size_type": "Cohen_dz",
        })
    else:
        stat, p = sps.wilcoxon(s, b)
        # 由 p 值反推 Z，再计算 r
        z = sps.norm.isf(p / 2)
        result.update({
            "test": "wilcoxon",
            "statistic": float(stat),
            "p_value": float(p),
            "effect_size": float(z / np.sqrt(n)),
            "effect_size_type": "r",
        })

    return result


def run_statistics(subject_metrics: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    对所有指标批量做 baseline vs stress 配对检验。

    Parameters
    ----------
    subject_metrics : DataFrame
        需含列 [subject, epoch, <各指标>]，epoch 取值 baseline / stress
    """
    cfg = config["statistics"]
    metrics = cfg.get("metrics", [])
    alpha = cfg.get("test_alpha", 0.05)

    wide = subject_metrics.pivot(index="subject", columns="epoch")

    rows = []
    for m in metrics:
        if m not in subject_metrics.columns:
            logger.warning("指标 %s 不存在，跳过。", m)
            continue
        try:
            b = wide[(m, "baseline")].to_numpy()
            s = wide[(m, "stress")].to_numpy()
        except KeyError:
            logger.warning("指标 %s 缺少 baseline 或 stress 分段，跳过。", m)
            continue
        rows.append(paired_test(b, s, m, config))

    if not rows:
        return pd.DataFrame()

    res = pd.DataFrame(rows)
    res["p_holm"] = _holm_bonferroni(res["p_value"].tolist())
    res["significant"] = res["p_holm"] < alpha
    res["significant_uncorrected"] = res["p_value"] < alpha

    # 数值列统一保留有效位数
    for c in ["baseline_mean", "baseline_sd", "stress_mean", "stress_sd",
              "mean_diff", "statistic", "effect_size"]:
        res[c] = res[c].round(4)
    for c in ["p_value", "p_holm", "normality_p"]:
        res[c] = res[c].round(5)

    return res
