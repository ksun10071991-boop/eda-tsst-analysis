"""
tools/calibrate_threshold.py
----------------------------
SCR 幅值阈值标定工具。

为什么需要标定:
  SCR 检测的 amplitude_min 直接决定检出的事件数。阈值过低会把噪声
  波动当作 SCR (假阳性)，过高会漏掉真实的弱反应。文献常用范围
  0.01-0.05 μS 跨度很大，最优值取决于信噪比、采样率与电极质量。

两种模式:
  --synthetic  在已知真值的合成信号上扫描阈值，找到还原真值最准的点。
               不需要任何真实数据，用于确定一个合理的起点。
  --real DIR   在真实 .acq 数据上扫描，输出每个阈值下的 SCR 频次。
               人类静息状态 NS-SCR 约 1-5 次/分钟，应激状态可达 10-20 次/分钟。
               选择使基线段落在生理合理区间的阈值。

用法:
    py tools/calibrate_threshold.py --synthetic
    py tools/calibrate_threshold.py --real data/raw
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from preprocess import preprocess      # noqa: E402
from decompose import decompose        # noqa: E402
from scr_detect import detect_scr      # noqa: E402

logging.getLogger().setLevel(logging.ERROR)   # 标定时静音日志

THRESHOLDS = [0.01, 0.02, 0.03, 0.04, 0.05, 0.08, 0.10]


def calibrate_synthetic(config: dict, fs: int = 50, duration: int = 300):
    """在合成信号上扫描阈值，对比检出数与真值。"""
    import neurokit2 as nk

    truths = [5, 10, 20, 40]
    print(f"\n合成信号标定 ({duration}s @ {fs}Hz，每个真值 3 次重复)\n")
    header = "amp_min | " + " | ".join(f"gt={t:<3d}" for t in truths) + " |  平均相对误差"
    print(header)
    print("-" * len(header))

    best, best_err = None, float("inf")
    for amp in THRESHOLDS:
        cfg = {**config, "scr_detection": {**config["scr_detection"], "amplitude_min": amp}}
        detected, errors = [], []
        for truth in truths:
            counts = []
            for seed in range(3):
                sig = nk.eda_simulate(duration=duration, sampling_rate=fs,
                                      scr_number=truth, drift=0.01, noise=0.01,
                                      random_state=seed)
                df = pd.DataFrame({"time_sec": np.arange(len(sig)) / fs, "eda_raw": sig})
                df.attrs["sampling_rate"] = float(fs)
                df = preprocess(df, cfg)
                df = decompose(df, cfg)
                counts.append(len(detect_scr(df, cfg)))
            mean_c = float(np.mean(counts))
            detected.append(mean_c)
            errors.append(abs(mean_c - truth) / truth)

        mean_err = float(np.mean(errors))
        if mean_err < best_err:
            best, best_err = amp, mean_err
        cells = " | ".join(f"{d:5.1f}" for d in detected)
        print(f"  {amp:.2f}  | {cells} |  {mean_err*100:6.1f}%")

    print(f"\n建议阈值: amplitude_min = {best}  (平均相对误差 {best_err*100:.1f}%)")
    print("把该值填入 config.yaml 的 scr_detection.amplitude_min\n")
    return best


def calibrate_real(config: dict, data_dir: Path):
    """在真实数据上扫描阈值，输出各阈值下的 SCR 频次供人工判断。"""
    from load_acq import load_eda

    files = sorted(data_dir.glob("*.acq"))
    if not files:
        print(f"在 {data_dir} 下未找到 .acq 文件。")
        return None

    print(f"\n真实数据标定 ({len(files)} 名被试)")
    print("参考区间: 静息 NS-SCR 约 1-5 次/分钟，应激状态约 10-20 次/分钟\n")

    header = "amp_min |  SCR 次/分钟 (均值 ± 标准差)  |  各被试范围"
    print(header)
    print("-" * len(header))

    rows = []
    for amp in THRESHOLDS:
        cfg = {**config, "scr_detection": {**config["scr_detection"], "amplitude_min": amp}}
        rates = []
        for f in files:
            try:
                df = load_eda(f, cfg)
                df = preprocess(df, cfg)
                df = decompose(df, cfg)
                scr = detect_scr(df, cfg)
                minutes = (df["time_sec"].iloc[-1] - df["time_sec"].iloc[0]) / 60.0
                rates.append(len(scr) / max(minutes, 1e-9))
            except Exception as exc:
                print(f"  [跳过 {f.name}: {exc}]")
        if not rates:
            continue
        m, sd = float(np.mean(rates)), float(np.std(rates))
        rows.append({"amplitude_min": amp, "mean_rate": m, "sd": sd})
        print(f"  {amp:.2f}  |      {m:5.2f} ± {sd:4.2f}          |  "
              f"{min(rates):.1f} – {max(rates):.1f}")

    if rows:
        res = pd.DataFrame(rows)
        # 挑最接近生理合理区间中值 (约 8 次/分钟，混合基线与应激) 的阈值
        res["dist"] = (res["mean_rate"] - 8.0).abs()
        best = res.loc[res["dist"].idxmin(), "amplitude_min"]
        print(f"\n建议起点: amplitude_min = {best}")
        print("请结合你的实验分段人工确认——基线段应落在 1-5 次/分钟。\n")
        return best
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="SCR 幅值阈值标定")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--synthetic", action="store_true", help="在合成信号上标定")
    g.add_argument("--real", metavar="DATA_DIR", help="在真实 .acq 数据上标定")
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if args.synthetic:
        calibrate_synthetic(config)
    else:
        calibrate_real(config, Path(args.real))
    return 0


if __name__ == "__main__":
    sys.exit(main())
