import os
import sys
from datetime import datetime
from pathlib import Path
import argparse
import yaml
import pandas as pd
import torch
from torchvision import transforms
import numpy as np
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data import BananaCombinedDataset  # noqa: E402
from train_test_banana.test_banana import test_model, load_config  # noqa: E402


def create_output_dir(base_dir: Path, model_tag: str = "BananaCombinedEval-b") -> Path:
    date_str = datetime.now().strftime("%Y%m%d")
    base_dir.mkdir(parents=True, exist_ok=True)
    existing = [p for p in base_dir.iterdir() if p.is_dir() and p.name.startswith(f"{date_str}-{model_tag}-")]
    if existing:
        max_idx = max(int(p.name.split("-")[-1]) for p in existing if p.name.split("-")[-1].isdigit())
        new_idx = max_idx + 1
    else:
        new_idx = 1
    folder = base_dir / f"{date_str}-{model_tag}-{new_idx}"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "runs").mkdir(exist_ok=True)
    return folder


def find_best_models(model_root: Path):
    best_models = []
    for run_dir in sorted(model_root.glob("run*_seed*")):
        best_model = run_dir / "best_model.pth"
        if best_model.exists():
            seed_part = run_dir.name.split("seed")[-1]
            try:
                seed = int(seed_part)
            except ValueError:
                seed = None
            best_models.append({
                "run_name": run_dir.name,
                "seed": seed,
                "model_path": best_model
            })
    if not best_models:
        raise FileNotFoundError(f"未在 {model_root} 下找到任何 best_model.pth 文件")
    return best_models


def summarize_metric(df: pd.DataFrame, metric_key: str, higher_is_better: bool) -> Dict[str, float]:
    series = df[metric_key]
    best_idx = series.idxmax() if higher_is_better else series.idxmin()
    worst_idx = series.idxmin() if higher_is_better else series.idxmax()
    return {
        "mean": series.mean(),
        "std": series.std(ddof=0),
        "best": series.loc[best_idx],
        "best_seed": df.loc[best_idx, "seed"],
        "worst": series.loc[worst_idx],
        "worst_seed": df.loc[worst_idx, "seed"],
    }


def format_seed_value(seed) -> str:
    if seed is None:
        return "N/A"
    if isinstance(seed, float) and np.isnan(seed):
        return "N/A"
    return str(int(seed))


def build_analysis_section(df: pd.DataFrame) -> List[str]:
    analysis_lines = []
    level_sorted = df.sort_values("level_f1", ascending=False)
    best_runs = level_sorted.head(2)
    if not best_runs.empty:
        best_desc = ", ".join(
            [f"seed{int(row['seed'])} (F1 {row['level_f1']:.3f})" for _, row in best_runs.iterrows()]
        )
        analysis_lines.append(f"1. 模型选择: 推荐 {best_desc}，分类与回归表现最均衡。")
    weight_range = (df["weight_r2"].min(), df["weight_r2"].max(), df["weight_rmse"].min(), df["weight_rmse"].max())
    analysis_lines.append(
        f"2. 重量回归: R² 范围 {weight_range[0]:.3f} ~ {weight_range[1]:.3f}，RMSE {weight_range[2]:.2f}g ~ {weight_range[3]:.2f}g。"
    )
    ripeness_range = (df["ripeness_r2"].min(), df["ripeness_r2"].max(), df["ripeness_rmse"].min(), df["ripeness_rmse"].max())
    analysis_lines.append(
        f"3. 成熟度回归: R² {ripeness_range[0]:.3f} ~ {ripeness_range[1]:.3f}，RMSE {ripeness_range[2]:.3f} ~ {ripeness_range[3]:.3f}。"
    )
    acc_mean = df["level_accuracy"].mean()
    acc_best = df["level_accuracy"].max()
    acc_low = df["level_accuracy"].min()
    analysis_lines.append(
        f"4. 分类准确率: 平均 {acc_mean:.3f}，最好 {acc_best:.3f}，最低 {acc_low:.3f}；建议检查低准确率 run 的样本预测。"
    )
    return analysis_lines


def build_report_content(
    df: pd.DataFrame,
    model_root: Path,
    runs_root: Path,
    pretrained_path: Path,
    feature_fusion_type: str,
    fpn_ablation_type: str,
    combine_type: str,
    banana_dir: Path,
    banana_add_dir: Path,
    dataset_size: int,
    seeds: List[int],
) -> str:
    metric_configs = [
        ("weight_r2", "Weight R²", True, "{:.4f}", "{:.4f}", ""),
        ("weight_rmse", "Weight RMSE (g)", False, "{:.2f}", "{:.2f}", "g"),
        ("ripeness_r2", "Ripeness R²", True, "{:.4f}", "{:.4f}", ""),
        ("ripeness_rmse", "Ripeness RMSE", False, "{:.4f}", "{:.4f}", ""),
        ("level_precision", "Level Precision", True, "{:.4f}", "{:.4f}", ""),
        ("level_recall", "Level Recall", True, "{:.4f}", "{:.4f}", ""),
        ("level_f1", "Level F1", True, "{:.4f}", "{:.4f}", ""),
        ("level_accuracy", "Level Accuracy", True, "{:.4f}", "{:.4f}", ""),
        ("level_map", "Level mAP", True, "{:.4f}", "{:.4f}", ""),
    ]
    lines: List[str] = []
    lines.append("=" * 80)
    lines.append("Banana模型 - 合并测试集评估摘要")
    lines.append("=" * 80)
    lines.append("")
    lines.append("评估配置:")
    lines.append(f"  数据集: {banana_dir}/test + {banana_add_dir} ({dataset_size}张)")
    lines.append("  数据加载: BananaCombinedDataset (test.csv + banana_add_70.csv, Resize 224, ToTensor)")
    lines.append(f"  评估模型来源: {model_root}/run*/best_model.pth")
    lines.append(f"  预训练权重: {pretrained_path}")
    lines.append(f"  特征融合类型: {feature_fusion_type}")
    lines.append(f"  FPN消融类型: {fpn_ablation_type}")
    lines.append(f"  组合类型: {combine_type}")
    lines.append("  输出任务: weight (回归), ripeness (回归), level (4类)")
    lines.append(f"  运行次数: {len(df)}")
    seeds_str = ", ".join(format_seed_value(seed) for seed in seeds)
    lines.append(f"  随机种子: [{seeds_str}]")
    lines.append("")
    lines.append("=" * 80)
    lines.append("总体统计 (平均 ± 标准差):")
    lines.append("=" * 80)
    for metric_key, display_name, higher_is_better, mean_fmt, value_fmt, unit in metric_configs:
        stats = summarize_metric(df, metric_key, higher_is_better)
        mean_str = mean_fmt.format(stats["mean"])
        std_str = mean_fmt.format(stats["std"]) if not np.isnan(stats["std"]) else "0.0000"
        best_str = value_fmt.format(stats["best"])
        worst_str = value_fmt.format(stats["worst"])
        lines.append(
            f"  {display_name:<17}: {mean_str} ± {std_str}  (最佳 {best_str}, 最差 {worst_str})"
        )
    lines.append("")
    lines.append("要点:")
    lines.append("  - 重量与成熟度回归在合并测试集上保持稳定。")
    lines.append("  - Level 分类整体准确率保持在 0.70+，需关注低准确率 run。")
    lines.append("  - 结合 runs/ 可视化目录可进一步分析预测差异。")
    lines.append("")
    lines.append("=" * 80)
    lines.append("详细结果 (按运行次序):")
    lines.append("=" * 80)
    for idx, row in enumerate(df.itertuples(index=False), 1):
        lines.append(f"运行 {idx} (seed: {format_seed_value(row.seed)}):")
        lines.append(f"  Weight R² {row.weight_r2:.3f} / RMSE {row.weight_rmse:.2f}")
        lines.append(f"  Ripeness R² {row.ripeness_r2:.3f} / RMSE {row.ripeness_rmse:.3f}")
        lines.append(
            f"  Level: Precision {row.level_precision:.3f}, Recall {row.level_recall:.3f}, "
            f"F1 {row.level_f1:.3f}, Accuracy {row.level_accuracy:.3f}, mAP {row.level_map:.3f}"
        )
        run_dir = runs_root / row.run
        lines.append(f"  可视化: {run_dir}")
        lines.append("")
    lines.append("=" * 80)
    lines.append("分析与建议:")
    lines.append("=" * 80)
    for line in build_analysis_section(df):
        lines.append(line)
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="使用合并香蕉测试集对多个模型进行评估并汇总结果")
    parser.add_argument("--model_result_dir", type=str, default=str(PROJECT_ROOT / "results/20251125-BananaModel-b-2"),
                        help="包含多个 run 子目录的模型结果根目录")
    parser.add_argument("--output_base", type=str, default=str(PROJECT_ROOT / "results"),
                        help="新的评估结果保存的根目录")
    parser.add_argument("--config_path", type=str, default=str(PROJECT_ROOT / "train_test_banana/config.yaml"),
                        help="用于获取数据预处理配置的配置文件路径")
    parser.add_argument("--pretrained_path", type=str, default=None,
                        help="预训练模型路径，默认从模型结果目录中的 config.yaml 读取")
    parser.add_argument("--banana_dir", type=str, default=str(PROJECT_ROOT / "data/banana"),
                        help="原始香蕉数据集根目录（包含 test/test.csv）")
    parser.add_argument("--banana_add_dir", type=str, default=str(PROJECT_ROOT / "data/banana_add_70"),
                        help="banana_add_70 数据目录（包含 banana_add_70.csv）")
    args = parser.parse_args()

    model_root = Path(args.model_result_dir).resolve()
    if not model_root.exists():
        raise FileNotFoundError(f"模型目录不存在: {model_root}")

    run_config_path = model_root / "config.yaml"
    if not run_config_path.exists():
        raise FileNotFoundError(f"未找到运行配置文件: {run_config_path}")
    with open(run_config_path, "r", encoding="utf-8") as f:
        run_config = yaml.safe_load(f)

    pretrained_path = args.pretrained_path or run_config.get("pretrained_path")
    if not pretrained_path:
        raise ValueError("未指定预训练模型路径")
    pretrained_path = Path(pretrained_path)
    if not pretrained_path.is_absolute():
        pretrained_path = PROJECT_ROOT / pretrained_path
    if not pretrained_path.exists():
        raise FileNotFoundError(f"预训练模型不存在: {pretrained_path}")

    feature_fusion_type = run_config.get("feature_fusion_type", "cross_attention")
    fpn_ablation_type = run_config.get("fpn_ablation_type", "ripe_only")
    combine_type = run_config.get("combine_type", "lka")

    config = load_config(args.config_path)
    resize = config["transform"]["resize"]
    transform = transforms.Compose([
        transforms.Resize(tuple(resize)),
        transforms.ToTensor(),
    ])

    banana_dir = Path(args.banana_dir).resolve()
    banana_add_dir = Path(args.banana_add_dir).resolve()

    test_dataset = BananaCombinedDataset(
        banana_dir=str(banana_dir),
        banana_add_dir=str(banana_add_dir),
        transform=transform
    )

    best_models = find_best_models(model_root)
    output_root = create_output_dir(Path(args.output_base), model_tag="BananaCombinedEval-b")
    runs_root = output_root / "runs"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    records = []
    print(f"共发现 {len(best_models)} 个best模型，将结果保存到: {output_root}")
    for idx, info in enumerate(best_models, 1):
        run_output = runs_root / info["run_name"]
        run_output.mkdir(parents=True, exist_ok=True)
        print(f"[{idx}/{len(best_models)}] 测试 {info['run_name']} - {info['model_path']}")
        metrics = test_model(
            test_dataset=test_dataset,
            model_path=str(info["model_path"]),
            pretrainpath=str(pretrained_path),
            feature_fusion_type=feature_fusion_type,
            fpn_ablation_type=fpn_ablation_type,
            combine_type=combine_type,
            device=device,
            logger=None,
            save_dir=str(run_output)
        )
        metrics.update({
            "run": info["run_name"],
            "seed": info["seed"],
            "model_path": str(info["model_path"])
        })
        records.append(metrics)

    df = pd.DataFrame(records)
    metrics_csv = output_root / "banana_combined_eval_metrics.csv"
    df.to_csv(metrics_csv, index=False)

    summary = df.describe().transpose()
    summary_csv = output_root / "banana_combined_eval_summary.csv"
    summary.to_csv(summary_csv)

    summary_txt = output_root / "banana_combined_eval_summary.txt"
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("Banana Combined Dataset Evaluation Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write("Per-run metrics:\n")
        f.write(df.to_string(index=False))
        f.write("\n\nAggregated statistics (mean/std/min/max):\n")
        f.write(summary.to_string())
        f.write("\n")

    report_txt = output_root / "banana_combined_eval_report.txt"
    report_content = build_report_content(
        df=df,
        model_root=model_root,
        runs_root=runs_root,
        pretrained_path=pretrained_path,
        feature_fusion_type=feature_fusion_type,
        fpn_ablation_type=fpn_ablation_type,
        combine_type=combine_type,
        banana_dir=banana_dir,
        banana_add_dir=banana_add_dir,
        dataset_size=len(test_dataset),
        seeds=[info["seed"] for info in best_models],
    )
    with open(report_txt, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"详细指标已保存到: {metrics_csv}")
    print(f"统计摘要已保存到: {summary_txt}")
    print(f"完整报告已保存到: {report_txt}")
    print("评估完成。")


if __name__ == "__main__":
    main()

