"""
run_analysis.py
---------------
主入口: 批量处理 .acq 文件 -> 预处理 -> SCL/SCR 分解 -> SCR 检测
        -> 分段指标提取 -> 配对统计 -> 图表与报告输出

用法:
    # 第一次拿到数据，先确认通道名
    py run_analysis.py --inspect data/raw/S01.acq

    # 批量分析
    py run_analysis.py --data-dir data/raw

    # 指定配置文件
    py run_analysis.py --data-dir data/raw --config config.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent / "src"))

from load_acq import load_eda, list_channels          # noqa: E402
from preprocess import preprocess                      # noqa: E402
from decompose import decompose                        # noqa: E402
from scr_detect import detect_scr, extract_all_epochs  # noqa: E402
from stats import run_statistics                       # noqa: E402
import visualize                                       # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("eda")


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_epoch_overrides(data_dir: Path) -> dict | None:
    """
    若存在 data/epochs.csv，按被试覆盖分段时间。
    格式: subject,epoch,start,end
    """
    p = data_dir.parent / "epochs.csv"
    if not p.exists():
        p = data_dir / "epochs.csv"
    if not p.exists():
        return None

    df = pd.read_csv(p)
    logger.info("使用 %s 的被试特定分段时间", p)
    out = {}
    for subj, grp in df.groupby("subject"):
        out[str(subj)] = {
            r["epoch"]: {"start": float(r["start"]), "end": float(r["end"])}
            for _, r in grp.iterrows()
        }
    return out


def process_subject(acq_path: Path, config: dict, epochs_cfg: dict,
                    out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """处理单个被试，返回 (分段指标, SCR事件表)。"""
    subject = acq_path.stem
    logger.info("=" * 62)
    logger.info("处理被试: %s", subject)

    try:
        df = load_eda(acq_path, config)
        df = preprocess(df, config)
        df = decompose(df, config)
        scr = detect_scr(df, config)
    except Exception as exc:
        logger.error("被试 %s 处理失败: %s", subject, exc)
        return None

    metrics = extract_all_epochs(df, scr, epochs_cfg)
    metrics.insert(0, "subject", subject)

    if config["output"].get("save_figures", True):
        fig_dir = out_dir / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        try:
            visualize.plot_subject_overview(
                df, scr, epochs_cfg, subject,
                fig_dir / f"{subject}_overview.png",
                config["output"].get("figure_dpi", 150))
        except Exception as exc:
            logger.warning("被试 %s 绘图失败: %s", subject, exc)

    # 保存逐被试中间结果，便于复查
    per_subj = out_dir / "per_subject"
    per_subj.mkdir(parents=True, exist_ok=True)
    scr.to_csv(per_subj / f"{subject}_scr_events.csv", index=False, encoding="utf-8-sig")

    scr = scr.copy()
    scr.insert(0, "subject", subject)
    return metrics, scr


def main() -> int:
    ap = argparse.ArgumentParser(description="EDA / TSST-C 分析管线")
    ap.add_argument("--data-dir", default="data/raw", help=".acq 文件所在目录")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--inspect", metavar="ACQ_FILE",
                    help="仅列出该 .acq 文件的通道信息后退出")
    args = ap.parse_args()

    if args.inspect:
        try:
            print(list_channels(args.inspect).to_string(index=False))
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            print(f"错误: {exc}")
            return 1
        print("\n把上面 EDA 通道的 name 或 index 填入 config.yaml。")
        return 0

    config = load_config(args.config)
    data_dir = Path(args.data_dir)
    out_dir = Path(config["output"].get("results_dir", "results"))
    out_dir.mkdir(parents=True, exist_ok=True)

    if not data_dir.exists():
        logger.error("目录不存在: %s", data_dir)
        return 1

    files = sorted(data_dir.glob("*.acq"))
    if not files:
        logger.error("在 %s 下未找到 .acq 文件。", data_dir)
        return 1
    logger.info("找到 %d 个 .acq 文件", len(files))

    default_epochs = config["epochs"]
    overrides = load_epoch_overrides(data_dir)

    all_metrics, all_scr = [], []
    for f in files:
        epochs_cfg = (overrides or {}).get(f.stem, default_epochs)
        result = process_subject(f, config, epochs_cfg, out_dir)
        if result is None:
            continue
        m, s = result
        all_metrics.append(m)
        if len(s):
            all_scr.append(s)

    if not all_metrics:
        logger.error("没有被试处理成功。")
        return 1

    subject_metrics = pd.concat(all_metrics, ignore_index=True)
    subject_metrics.to_csv(out_dir / "subject_metrics.csv",
                           index=False, encoding="utf-8-sig")

    if all_scr:
        pd.concat(all_scr, ignore_index=True).to_csv(
            out_dir / "all_scr_events.csv", index=False, encoding="utf-8-sig")

    logger.info("=" * 62)
    logger.info("运行统计检验 (baseline vs stress)")
    results = run_statistics(subject_metrics, config)

    if results.empty:
        logger.warning("无可用统计结果。")
        return 0

    results.to_csv(out_dir / "statistics.csv", index=False, encoding="utf-8-sig")

    if config["output"].get("save_figures", True):
        fig_dir = out_dir / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        dpi = config["output"].get("figure_dpi", 150)
        try:
            visualize.plot_paired_comparison(
                subject_metrics, results, fig_dir / "group_paired_comparison.png", dpi)
            visualize.plot_effect_sizes(
                results, fig_dir / "effect_sizes.png", dpi)
        except Exception as exc:
            logger.warning("组水平绘图失败: %s", exc)

    print("\n" + "=" * 78)
    print("统计结果摘要 (baseline vs stress)")
    print("=" * 78)
    cols = ["metric", "n", "baseline_mean", "stress_mean", "mean_diff",
            "test", "p_value", "p_holm", "effect_size", "significant"]
    print(results[cols].to_string(index=False))
    print("=" * 78)
    print(f"\n完整结果已保存至: {out_dir.resolve()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
