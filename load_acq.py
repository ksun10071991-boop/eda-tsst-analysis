"""
src/load_acq.py
-----------
读取 BIOPAC AcqKnowledge (.acq) 原始文件，定位 EDA 通道并降采样。

依赖: bioread (纯 Python 读取 .acq，无需 AcqKnowledge 软件)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.signal import decimate

logger = logging.getLogger(__name__)


def list_channels(acq_path: str | Path) -> pd.DataFrame:
    """
    列出 .acq 文件中所有通道的名称、单位和采样率。
    第一次拿到数据时先跑这个，确认 EDA 通道叫什么。
    """
    import bioread

    acq_path = Path(acq_path)
    if not acq_path.exists():
        raise FileNotFoundError(f"文件不存在: {acq_path}")
    if acq_path.suffix.lower() != ".acq":
        raise ValueError(
            f"'{acq_path.name}' 不是 .acq 文件 (后缀为 '{acq_path.suffix}')。"
            f"--inspect 只接受 BIOPAC AcqKnowledge 格式。"
        )

    try:
        data = bioread.read_file(str(acq_path))
    except Exception as exc:
        raise RuntimeError(
            f"无法读取 '{acq_path.name}': {exc}\n"
            f"请确认该文件是有效的 BIOPAC .acq 记录且未损坏。"
        ) from exc

    rows = []
    for i, ch in enumerate(data.channels):
        rows.append({
            "index": i,
            "name": ch.name,
            "units": ch.units,
            "samples": len(ch.data),
            "sampling_rate": ch.samples_per_second,
        })
    return pd.DataFrame(rows)


def _resolve_channel(channels, name: Optional[str], index: Optional[int],
                     keywords: list[str]):
    """按 index → name → 关键词模糊匹配 的优先级定位 EDA 通道。"""
    if index is not None:
        logger.info("按索引定位通道: %d (%s)", index, channels[index].name)
        return channels[index]

    if name is not None:
        for ch in channels:
            if ch.name.strip().lower() == name.strip().lower():
                logger.info("按名称定位通道: %s", ch.name)
                return ch
        raise ValueError(f"未找到名为 '{name}' 的通道。"
                         f"可用通道: {[c.name for c in channels]}")

    # 关键词模糊匹配
    for ch in channels:
        lowered = ch.name.lower()
        if any(kw.lower() in lowered for kw in keywords):
            logger.info("按关键词自动定位通道: %s", ch.name)
            return ch

    raise ValueError(
        f"无法自动定位 EDA 通道。请在 config.yaml 中显式指定 "
        f"eda_channel_name 或 eda_channel_index。"
        f"可用通道: {[c.name for c in channels]}"
    )


def _downsample(signal: np.ndarray, orig_fs: float, target_fs: float) -> tuple[np.ndarray, float]:
    """
    整数倍降采样。decimate 内置抗混叠低通滤波，比直接切片安全。
    非整数倍时退化为 FFT 重采样。
    """
    if target_fs >= orig_fs:
        return signal, orig_fs

    factor = orig_fs / target_fs
    if abs(factor - round(factor)) < 1e-9:
        factor = int(round(factor))
        # decimate 单次因子过大会不稳定，分解为多级
        out = signal
        remaining = factor
        while remaining > 1:
            step = min(remaining, 10)
            while remaining % step != 0 and step > 1:
                step -= 1
            if step == 1:
                break
            out = decimate(out, step, ftype="iir", zero_phase=True)
            remaining //= step
        actual_fs = orig_fs / (factor // max(remaining, 1))
        return out, orig_fs / factor
    else:
        from scipy.signal import resample
        n_out = int(len(signal) * target_fs / orig_fs)
        return resample(signal, n_out), target_fs


def load_eda(acq_path: str | Path, config: dict) -> pd.DataFrame:
    """
    读取 .acq 并返回 DataFrame: [time_sec, eda_raw]

    Returns
    -------
    df : pd.DataFrame
        含 attrs['sampling_rate'] 与 attrs['channel_name']
    """
    import bioread

    acq_cfg = config["acquisition"]
    acq_path = Path(acq_path)
    if not acq_path.exists():
        raise FileNotFoundError(f"文件不存在: {acq_path}")

    logger.info("读取 %s", acq_path.name)
    data = bioread.read_file(str(acq_path))

    ch = _resolve_channel(
        data.channels,
        acq_cfg.get("eda_channel_name"),
        acq_cfg.get("eda_channel_index"),
        acq_cfg.get("eda_channel_keywords", ["eda", "gsr"]),
    )

    raw = np.asarray(ch.data, dtype=float)
    orig_fs = float(ch.samples_per_second)

    # 以文件实际采样率为准，config 中的值仅作校验
    declared = acq_cfg.get("original_sampling_rate")
    if declared and abs(declared - orig_fs) > 1:
        logger.warning("config 声明采样率 %.1f Hz，文件实际为 %.1f Hz，以文件为准。",
                       declared, orig_fs)

    target_fs = float(acq_cfg.get("target_sampling_rate", orig_fs))
    signal, fs = _downsample(raw, orig_fs, target_fs)

    logger.info("通道 '%s' | %.1f Hz -> %.1f Hz | 时长 %.1f 秒",
                ch.name, orig_fs, fs, len(signal) / fs)

    df = pd.DataFrame({
        "time_sec": np.arange(len(signal)) / fs,
        "eda_raw": signal,
    })
    df.attrs["sampling_rate"] = fs
    df.attrs["channel_name"] = ch.name
    df.attrs["units"] = ch.units
    df.attrs["source_file"] = acq_path.name
    return df
