"""
scr_detect.py
-------------
SCR 事件检测与分段指标提取。

SCR 事件的定义参数:
  amplitude  —— 波峰值减去起始点值 (μS)。低于阈值(常用 0.01-0.05 μS)
                的波动视为噪声不计入。
  rise time  —— 起始点到波峰的时间，生理典型值 1-3 秒
  recovery   —— 波峰回落到 50% 幅值所需时间

分段指标 (baseline vs stress 比较用):
  SCL_mean            平均皮电水平
  SCL_slope           皮电水平线性趋势斜率，反映唤醒的持续上升/下降
  SCR_count           该段内 SCR 事件总数
  SCR_rate_per_min    每分钟 SCR 频次 —— 段长不等时必须用这个而非 count
  SCR_amplitude_mean  平均幅值
  SCR_amplitude_sum   幅值总和 (反映总体交感激活量)
  EDA_std             信号波动性
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def detect_scr(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    在相位性成分上检测 SCR 峰值。

    Returns
    -------
    scr_df : pd.DataFrame
        每行一个 SCR 事件: [onset_sec, peak_sec, amplitude, rise_time_sec]
    """
    import neurokit2 as nk

    cfg = config["scr_detection"]
    fs = df.attrs["sampling_rate"]
    phasic = df["EDA_Phasic"].to_numpy(dtype=float)
    amp_min = cfg.get("amplitude_min", 0.01)

    try:
        _, info = nk.eda_peaks(
            phasic,
            sampling_rate=fs,
            method=cfg.get("method", "neurokit"),
            amplitude_min=amp_min,
        )
    except Exception as exc:
        logger.error("SCR 检测失败: %s", exc)
        return pd.DataFrame(columns=["onset_sec", "peak_sec", "amplitude", "rise_time_sec"])

    peaks = np.asarray(info.get("SCR_Peaks", []), dtype=float)
    onsets = np.asarray(info.get("SCR_Onsets", []), dtype=float)
    amps = np.asarray(info.get("SCR_Amplitude", []), dtype=float)
    rise = np.asarray(info.get("SCR_RiseTime", []), dtype=float)

    n = len(peaks)
    if n == 0:
        logger.info("未检测到 SCR 事件。")
        return pd.DataFrame(columns=["onset_sec", "peak_sec", "amplitude", "rise_time_sec"])

    def _pad(arr):
        arr = np.asarray(arr, dtype=float)
        if len(arr) == n:
            return arr
        out = np.full(n, np.nan)
        out[:min(len(arr), n)] = arr[:min(len(arr), n)]
        return out

    scr = pd.DataFrame({
        "onset_sec": _pad(onsets) / fs,
        "peak_sec": peaks / fs,
        "amplitude": _pad(amps),
        "rise_time_sec": _pad(rise),
    })

    # 幅值阈值过滤 (双保险，某些方法不严格执行 amplitude_min)
    before = len(scr)
    scr = scr[scr["amplitude"].fillna(0) >= amp_min].reset_index(drop=True)
    if before != len(scr):
        logger.info("幅值阈值过滤: %d -> %d 个事件", before, len(scr))

    logger.info("检测到 %d 个 SCR 事件，平均幅值 %.4f μS",
                len(scr), scr["amplitude"].mean() if len(scr) else 0.0)
    return scr


def _linear_slope(t: np.ndarray, y: np.ndarray) -> float:
    """最小二乘线性斜率 (μS/min)。"""
    if len(t) < 2:
        return np.nan
    coef = np.polyfit(t, y, 1)[0]
    return float(coef * 60.0)


def extract_epoch_metrics(df: pd.DataFrame, scr: pd.DataFrame,
                          start: float, end: float, label: str) -> dict:
    """提取单个实验分段 [start, end) 内的所有指标。"""
    seg = df[(df["time_sec"] >= start) & (df["time_sec"] < end)]
    if len(seg) == 0:
        logger.warning("分段 '%s' [%.0f-%.0fs] 无数据。", label, start, end)
        return {"epoch": label}

    duration_min = (seg["time_sec"].iloc[-1] - seg["time_sec"].iloc[0]) / 60.0
    duration_min = max(duration_min, 1e-9)

    # SCR 事件按峰值时刻归属分段
    ev = scr[(scr["peak_sec"] >= start) & (scr["peak_sec"] < end)] if len(scr) else scr

    tonic = seg["EDA_Tonic"].to_numpy()
    t = seg["time_sec"].to_numpy()

    return {
        "epoch": label,
        "duration_sec": round(float(seg["time_sec"].iloc[-1] - seg["time_sec"].iloc[0]), 2),
        "artifact_pct": round(float(100 * seg["artifact"].mean()), 2),
        "SCL_mean": float(np.nanmean(tonic)),
        "SCL_slope": _linear_slope(t, tonic),
        "SCR_count": int(len(ev)),
        "SCR_rate_per_min": float(len(ev) / duration_min),
        "SCR_amplitude_mean": float(ev["amplitude"].mean()) if len(ev) else 0.0,
        "SCR_amplitude_sum": float(ev["amplitude"].sum()) if len(ev) else 0.0,
        "EDA_std": float(np.nanstd(seg["eda_clean"].to_numpy())),
    }


def extract_all_epochs(df: pd.DataFrame, scr: pd.DataFrame,
                       epochs_cfg: dict) -> pd.DataFrame:
    """按配置提取 baseline 与 stress 两段的指标。"""
    rows = []
    for label, window in epochs_cfg.items():
        rows.append(extract_epoch_metrics(
            df, scr, float(window["start"]), float(window["end"]), label))
    return pd.DataFrame(rows)
