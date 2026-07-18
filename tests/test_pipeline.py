"""
tests/test_pipeline.py
----------------------
用 NeuroKit2 合成的 EDA 信号验证管线正确性，无需真实 .acq 文件。
合成数据中 stress 段人为设置更高的 SCR 频次与 SCL 水平，
管线应当能检测出显著差异。

运行: py tests/test_pipeline.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import neurokit2 as nk
from preprocess import preprocess
from decompose import decompose
from scr_detect import detect_scr, extract_all_epochs
from stats import run_statistics


def make_subject(seed, fs=50, dur=300, scr_per_min_base=2, scr_per_min_stress=8,
                 scl_shift=1.5):
    """合成一名被试的 baseline + stress 连续记录。"""
    rng = np.random.default_rng(seed)

    base = nk.eda_simulate(duration=dur, sampling_rate=fs,
                           scr_number=int(scr_per_min_base * dur / 60),
                           drift=0.01, noise=0.01, random_state=seed)
    stress = nk.eda_simulate(duration=dur, sampling_rate=fs,
                             scr_number=int(scr_per_min_stress * dur / 60),
                             drift=0.05, noise=0.01, random_state=seed + 1000)

    # stress 段整体抬高 SCL，模拟应激唤醒
    stress = stress + scl_shift + rng.normal(0, 0.1)

    sig = np.concatenate([base, stress])
    df = pd.DataFrame({"time_sec": np.arange(len(sig)) / fs, "eda_raw": sig})
    df.attrs["sampling_rate"] = float(fs)
    return df


def main():
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    fs, dur = 50, 300
    config["epochs"] = {
        "baseline": {"start": 0, "end": dur},
        "stress": {"start": dur, "end": 2 * dur},
    }
    config["decomposition"]["method"] = "highpass"  # 测试用，快

    n_subjects = 12
    all_metrics = []

    print(f"合成 {n_subjects} 名被试数据并运行管线 ...\n")
    for i in range(n_subjects):
        df = make_subject(seed=42 + i, fs=fs, dur=dur)
        df = preprocess(df, config)
        df = decompose(df, config)
        scr = detect_scr(df, config)
        m = extract_all_epochs(df, scr, config["epochs"])
        m.insert(0, "subject", f"S{i+1:02d}")
        all_metrics.append(m)

        b = m[m.epoch == "baseline"].iloc[0]
        s = m[m.epoch == "stress"].iloc[0]
        print(f"  S{i+1:02d}  SCL {b.SCL_mean:6.2f} -> {s.SCL_mean:6.2f} μS   "
              f"SCR/min {b.SCR_rate_per_min:5.2f} -> {s.SCR_rate_per_min:5.2f}")

    subject_metrics = pd.concat(all_metrics, ignore_index=True)
    results = run_statistics(subject_metrics, config)

    print("\n" + "=" * 76)
    print("统计结果")
    print("=" * 76)
    cols = ["metric", "n", "baseline_mean", "stress_mean", "mean_diff",
            "test", "p_value", "p_holm", "effect_size", "significant"]
    print(results[cols].to_string(index=False))
    print("=" * 76)

    # 断言: 应激应当显著抬高 SCL 与 SCR 频次
    checks = []
    for metric in ["SCL_mean", "SCR_rate_per_min"]:
        row = results[results.metric == metric]
        if row.empty:
            checks.append((metric, False, "缺失"))
            continue
        row = row.iloc[0]
        ok = bool(row.significant) and row.mean_diff > 0
        checks.append((metric, ok, f"p_holm={row.p_holm}, diff={row.mean_diff}"))

    print("\n验证:")
    all_ok = True
    for name, ok, info in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {info}")
        all_ok &= ok

    print("\n管线验证" + ("通过 ✓" if all_ok else "未通过 ✗"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
