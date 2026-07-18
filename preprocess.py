"""
src/preprocess.py
-------------
EDA 信号预处理: 低通滤波 + 运动伪迹检测与处理。

原理说明 (面试可能会问):
  - EDA 是慢变信号，有效成分集中在 0-5 Hz。SCR 的上升时间约 1-3 秒，
    对应频率远低于 1 Hz。因此 1-3 Hz 低通即可滤除工频与运动高频噪声，
    同时完整保留 SCR 波形。
  - 用零相位滤波 (filtfilt) 而非单向滤波，避免引入相位延迟——
    这一点对后续 SCR 潜伏期 (latency) 的测量至关重要。
  - 运动伪迹表现为一阶差分上的尖峰: 真实 SCR 的上升速率有生理上限
    (约 < 1-2 μS/s)，超过该阈值的突变几乎一定是电极移动或压迫所致。
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

logger = logging.getLogger(__name__)


def lowpass_filter(signal: np.ndarray, fs: float, cutoff: float,
                   order: int = 4) -> np.ndarray:
    """零相位 Butterworth 低通滤波。"""
    nyq = fs / 2.0
    if cutoff >= nyq:
        logger.warning("截止频率 %.2f Hz >= Nyquist %.2f Hz，跳过滤波。", cutoff, nyq)
        return signal.copy()

    b, a = butter(order, cutoff / nyq, btype="low")
    # padlen 需小于信号长度
    padlen = min(3 * max(len(a), len(b)), len(signal) - 1)
    return filtfilt(b, a, signal, padlen=padlen)


def detect_artifacts(signal: np.ndarray, fs: float,
                     percentile: float = 99.5,
                     percentile_multiplier: float = 3.0,
                     absolute_ceiling: float | None = 15.0) -> np.ndarray:
    """
    基于一阶差分的运动伪迹检测 (自适应阈值)。

    设计过程 (两次修正，记录以备查):
      v1 固定阈值 2 uS/s -- 失败。实测真实 SCR 上升速率可达 6 uS/s，
         结果把每个正常 SCR 波峰都误标为伪迹。
      v2 median + k*MAD -- 仍失败。EDA 差分分布高度右偏 (绝大多数样本
         接近 0，少数 SCR 上升沿很大)，median 与 MAD 都被压在接近 0 处，
         阈值过低; 且 SCR 密集的片段会自己抬高误检率
         (实测密集 SCR 误检 14%，而真实电极跳变仅 5%，完全反了)。
      v3 高分位数缩放 (当前) -- 以 p99.5 作为该被试"正常信号动态范围"
         的稳健估计，再乘一个倍数作为阈值。SCR 上升沿即使很陡也落在
         p99.5 附近，而电极移动的阶跃比它快一个数量级，因此可分离。

    absolute_ceiling 作为兜底: 超过该速率在生理上不可能是 SCR。

    Returns
    -------
    mask : np.ndarray[bool]
        True 表示该样本点为伪迹。
    """
    # 差分转换为 uS/s
    derivative = np.abs(np.diff(signal, prepend=signal[0])) * fs

    ref = float(np.percentile(derivative, percentile))
    if ref <= 0:
        threshold = absolute_ceiling if absolute_ceiling else np.inf
    else:
        threshold = ref * percentile_multiplier
        if absolute_ceiling is not None:
            threshold = min(threshold, absolute_ceiling)

    logger.debug("伪迹阈值 %.3f uS/s (p%.1f=%.3f)", threshold, percentile, ref)
    mask = derivative > threshold

    # 伪迹通常成段出现，向两侧各扩展 0.5 秒以覆盖完整突变
    pad = int(0.25 * fs)
    if pad > 0 and mask.any():
        padded = mask.copy()
        idx = np.where(mask)[0]
        for i in idx:
            lo, hi = max(0, i - pad), min(len(mask), i + pad + 1)
            padded[lo:hi] = True
        mask = padded

    pct = 100 * mask.sum() / len(mask)
    if pct > 0:
        logger.info("检测到伪迹样本 %.2f%% (%d 点)", pct, int(mask.sum()))
    if pct > 20:
        logger.warning("伪迹比例超过 20%%，该记录质量存疑，建议人工检查。")

    return mask


def handle_artifacts(signal: np.ndarray, mask: np.ndarray,
                     method: str = "interpolate") -> np.ndarray:
    """对伪迹段做线性插值或置为 NaN。"""
    out = signal.astype(float).copy()

    if method == "nan":
        out[mask] = np.nan
        return out

    if not mask.any():
        return out

    if mask.all():
        logger.error("全部样本被标记为伪迹，返回原信号。")
        return signal.astype(float).copy()

    good_idx = np.where(~mask)[0]
    bad_idx = np.where(mask)[0]
    out[bad_idx] = np.interp(bad_idx, good_idx, signal[good_idx])
    return out


def preprocess(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    完整预处理流程。输入需含 'eda_raw'，输出增加 'eda_clean' 与 'artifact'。
    """
    cfg = config["preprocessing"]
    fs = df.attrs["sampling_rate"]
    raw = df["eda_raw"].to_numpy(dtype=float)

    # 1) 伪迹检测在滤波前做 —— 滤波会把尖峰抹平，事后就检测不到了
    mask = detect_artifacts(
        raw, fs,
        percentile=cfg.get("artifact_percentile", 99.5),
        percentile_multiplier=cfg.get("artifact_percentile_multiplier", 3.0),
        absolute_ceiling=cfg.get("artifact_absolute_ceiling_us_per_sec", 15.0),
    )

    # 2) 先修补伪迹，再滤波，避免伪迹在滤波时向邻域扩散
    repaired = handle_artifacts(raw, mask, cfg.get("artifact_handling", "interpolate"))

    # 3) 低通滤波
    clean = lowpass_filter(repaired, fs,
                           cfg.get("lowpass_cutoff", 3.0),
                           cfg.get("filter_order", 4))

    out = df.copy()
    out["eda_clean"] = clean
    out["artifact"] = mask
    out.attrs = dict(df.attrs)
    out.attrs["artifact_pct"] = float(100 * mask.sum() / len(mask))
    return out
