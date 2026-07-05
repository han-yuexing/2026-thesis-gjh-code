import os
import sys
from datetime import datetime
from pathlib import Path
import argparse
import yaml
import pandas as pd
import torch
from torchvision import transforms
from torch.utils.data import random_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data import BananaDataset  # noqa: E402
from train_test_banana.test_banana import test_model, load_config  # noqa: E402


def create_output_dir(base_dir: Path, model_tag: str = "BananaEvalValidation-b") -> Path:
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


def get_validation_subset(dataset, seed):
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(seed)
    _, val_subset = random_split(dataset, [train_size, val_size], generator=generator)
    return val_subset


def main():
    parser = argparse.ArgumentParser(description="在原始训练集的验证划分上评估多个模型并汇总结果")
    parser.add_argument("--model_result_dir", type=str, default=str(PROJECT_ROOT / "results/20251029-BananaModel-b-1"),
                        help="包含多个 run 子目录的模型结果根目录")
    parser.add_argument("--output_base", type=str, default=str(PROJECT_ROOT / "results"),
                        help="新的评估结果保存的根目录")
    parser.add_argument("--config_path", type=str, default=str(PROJECT_ROOT / "train_test_banana/config.yaml"),
                        help="用于获取数据预处理配置的配置文件路径")
    parser.add_argument("--pretrained_path", type=str, default=None,
                        help="预训练模型路径，默认从模型结果目录中的 config.yaml 读取")
    parser.add_argument("--banana_dir", type=str, default=str(PROJECT_ROOT / "data/banana"),
                        help="香蕉数据集根目录（包含 train/ 子目录）")
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

    eval_transform = transforms.Compose([
        transforms.Resize(tuple(resize)),
        transforms.ToTensor(),
    ])

    banana_dir = Path(args.banana_dir).resolve()
    train_dir = banana_dir / "train"

    best_models = find_best_models(model_root)
    output_root = create_output_dir(Path(args.output_base), model_tag="BananaEvalValidation-b")
    runs_root = output_root / "runs"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    records = []
    print(f"共发现 {len(best_models)} 个best模型，将结果保存到: {output_root}")
    for idx, info in enumerate(best_models, 1):
        run_output = runs_root / info["run_name"]
        run_output.mkdir(parents=True, exist_ok=True)

        dataset = BananaDataset(image_dir=str(train_dir), mode="train", transform=eval_transform)
        val_subset = get_validation_subset(dataset, info["seed"])

        print(f"[{idx}/{len(best_models)}] 测试 {info['run_name']} - {info['model_path']} (seed={info['seed']})")
        metrics = test_model(
            test_dataset=val_subset,
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
            "model_path": str(info["model_path"]),
            "val_size": len(val_subset)
        })
        records.append(metrics)

    df = pd.DataFrame(records)
    metrics_csv = output_root / "banana_validation_eval_metrics.csv"
    df.to_csv(metrics_csv, index=False)

    summary = df.describe().transpose()
    summary_csv = output_root / "banana_validation_eval_summary.csv"
    summary.to_csv(summary_csv)

    summary_txt = output_root / "banana_validation_eval_summary.txt"
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("Banana Validation Split Evaluation Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write("Per-run metrics:\n")
        f.write(df.to_string(index=False))
        f.write("\n\nAggregated statistics (mean/std/min/max):\n")
        f.write(summary.to_string())
        f.write("\n")

    print(f"详细指标已保存到: {metrics_csv}")
    print(f"统计摘要已保存到: {summary_txt}")
    print("评估完成。")


if __name__ == "__main__":
    main()

