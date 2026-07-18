"""
src/decompose.py
------------
将 EDA 分解为紧张性成分 (SCL, tonic) 与相位性成分 (SCR, phasic)。

概念区分 (面试高频问题):
  SCL (Skin Conductance Level, 紧张性/tonic)
    - 缓慢漂移的基线水平，时间尺度以分钟计
    - 反映整体唤醒水平 (general arousal)，个体差异极大
    - 因此组间比较通常看"变化量"而非绝对值

  SCR (Skin Conductance Response, 相位性/phasic)
    - 叠加在 SCL 之上的短时波动，上升时间 1-3 秒，恢复 3-10 秒
    - 分两类:
        ER-SCR (event-related): 刺激后 1-4 秒窗口内出现，可归因于特定事件
        NS-SCR (non-specific): 无明确刺激对应，其频次反映持续唤醒状态
    - TSST-C 这类持续性应激任务，主要看 NS-SCR 频次而非单次事件

分解方法:
  cvxEDA  —— 凸优化框架，把 EDA 建模为 tonic + phasic 卷积 + 噪声，
             通过求解带约束的二次规划分离。精度最高，速度较慢。
  highpass —— 简单高通滤波 (0.05 Hz) 分离，快但对慢速 SCR 有损失。
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def decompose(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    分解 EDA 为 SCL (tonic) 与 SCR (phasic)。

    输入需含 'eda_clean'，输出增加 'EDA_Tonic' (SCL) 与 'EDA_Phasic' (SCR)。
    """
    import neurokit2 as nk

    method = config["decomposition"].get("method", "cvxEDA")
    fs = df.attrs["sampling_rate"]
    signal = df["eda_clean"].to_numpy(dtype=float)

    if np.isnan(signal).any():
        logger.warning("信号含 NaN，分解前用前向填充处理。")
        signal = pd.Series(signal).ffill().bfill().to_numpy()

    logger.info("使用 %s 方法分解 SCL/SCR ...", method)

    try:
        components = nk.eda_phasic(signal, sampling_rate=fs, method=method)
    except Exception as exc:
        logger.error("%s 分解失败 (%s)，回退到 highpass 方法。", method, exc)
        components = nk.eda_phasic(signal, sampling_rate=fs, method="highpass")

    out = df.copy()
    out["EDA_Tonic"] = components["EDA_Tonic"].to_numpy()
    out["EDA_Phasic"] = components["EDA_Phasic"].to_numpy()
    out.attrs = dict(df.attrs)
    out.attrs["decomposition_method"] = method

    logger.info("SCL 均值 %.3f μS | SCR 成分标准差 %.4f μS",
                out["EDA_Tonic"].mean(), out["EDA_Phasic"].std())
    return out
