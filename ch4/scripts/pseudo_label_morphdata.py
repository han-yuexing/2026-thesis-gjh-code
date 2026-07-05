import argparse
import csv
import os
import sys
from dataclasses import dataclass
from typing import List, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


LEVEL_ID_TO_GRADE = {0: "super", 1: "first", 2: "second", 3: "third"}

# Ensure repo root is importable (so `import FruVeg26...` works)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


@dataclass(frozen=True)
class TaskSpec:
    name: str
    config_path: str
    default_image_subdir: str


TASKS = {
    "banana": TaskSpec(
        name="banana",
        config_path=os.path.join("FruVeg26", "train_test_banana", "config.yaml"),
        # 用户目前需求：对 masked 打伪标签
        default_image_subdir=os.path.join("morphdata", "banana", "masked"),
    ),
    "cucumber": TaskSpec(
        name="cucumber",
        config_path=os.path.join("FruVeg26", "train_test_cucumber", "config.yaml"),
        # 用户目前需求：对 masked 打伪标签
        default_image_subdir=os.path.join("morphdata", "cucumber", "masked"),
    ),
}


class ImagePathDataset(Dataset):
    def __init__(self, image_paths: List[str], tfm):
        self.image_paths = image_paths
        self.tfm = tfm

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        p = self.image_paths[idx]
        img = Image.open(p).convert("RGB")
        if self.tfm is not None:
            img = self.tfm(img)
        return img, p


def _load_yaml(path: str) -> dict:
    if yaml is None:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _list_images(morph_dir: str) -> List[str]:
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
    out = []
    for name in os.listdir(morph_dir):
        if name.lower().endswith(exts):
            out.append(os.path.join(morph_dir, name))
    out.sort()
    return out


def _build_transform(cfg: dict):
    resize_hw = (cfg.get("transform", {}) or {}).get("resize", [224, 224])
    return transforms.Compose(
        [
            transforms.Resize(tuple(resize_hw)),
            transforms.ToTensor(),
        ]
    )


def _load_model(task: str, model_path: str, pretrained_path: str, device: torch.device, cfg: dict):
    if task == "banana":
        from FruVeg26.model.banana import BananaModel

        model = BananaModel(
            pretrained_path,
            feature_fusion_type=cfg["model"]["feature_fusion_type"],
            fpn_ablation_type=cfg["model"]["fpn_ablation_type"],
            combine_type=cfg["model"]["combine_type"],
        )
        state_dict = torch.load(model_path, map_location="cpu")
        load_info = model.load_state_dict(state_dict, strict=False)
        missing = getattr(load_info, "missing_keys", [])
        unexpected = getattr(load_info, "unexpected_keys", [])
        if missing or unexpected:
            print(f"[warn] load_state_dict strict=False missing={len(missing)} unexpected={len(unexpected)}")
        model.to(device).eval()
        return model

    if task == "cucumber":
        from FruVeg26.model.cucumber import CucumberModel

        model = CucumberModel(
            pretrained_path,
            feature_fusion_type=cfg["model"]["feature_fusion_type"],
            fpn_ablation_type=cfg["model"]["fpn_ablation_type"],
            combine_type=cfg["model"]["combine_type"],
        )
        state_dict = torch.load(model_path, map_location="cpu")
        load_info = model.load_state_dict(state_dict, strict=False)
        missing = getattr(load_info, "missing_keys", [])
        unexpected = getattr(load_info, "unexpected_keys", [])
        if missing or unexpected:
            print(f"[warn] load_state_dict strict=False missing={len(missing)} unexpected={len(unexpected)}")
        model.to(device).eval()
        return model

    raise ValueError(f"Unknown task: {task}")


@torch.no_grad()
def _infer(task: str, model, loader: DataLoader, device: torch.device):
    rows = []
    for images, paths in loader:
        images = images.to(device, non_blocking=True)
        out1, out2, logits = model(images)

        weight = out1.detach().float().squeeze(-1).cpu().numpy().tolist()
        aux = out2.detach().float().squeeze(-1).cpu().numpy().tolist()
        pred_grade_id = torch.argmax(logits.detach(), dim=1).cpu().numpy().tolist()
        pred_grade = [LEVEL_ID_TO_GRADE[int(i)] for i in pred_grade_id]

        for p, w, a, g in zip(paths, weight, aux, pred_grade):
            if task == "banana":
                rows.append({"image": p, "weight": float(w), "ripeness": float(a), "grade": g})
            else:
                rows.append({"image": p, "weight": float(w), "arch": float(a), "grade": g})
    return rows


def main():
    parser = argparse.ArgumentParser(description="用训练好的 pth 对 morphdata 打伪标签并输出 CSV")
    parser.add_argument("--task", type=str, required=True, choices=sorted(TASKS.keys()))
    parser.add_argument("--model_path", type=str, required=True, help="训练好的 .pth 路径")
    parser.add_argument(
        "--image_dir",
        type=str,
        default=None,
        help="要打标签的图片目录（默认 morphdata/<task>/masked）",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="对应任务的 config.yaml（默认使用 FruVeg26/train_test_*/config.yaml）",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="backbone 预训练权重路径（默认取 config.model.pretrained_path）",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--banana_weight_only",
        action="store_true",
        help="香蕉输出仅保留 weight 一列（不包含 image/ripeness/grade）",
    )
    parser.add_argument("--out_csv", type=str, default=None, help="输出 CSV 路径（默认 <image_dir>/pseudo_labels.csv）")
    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    spec = TASKS[args.task]
    cfg_path = args.config or os.path.join(repo_root, spec.config_path)
    cfg = _load_yaml(cfg_path)

    image_dir = args.image_dir or os.path.join(repo_root, spec.default_image_subdir)
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"image_dir not found: {image_dir}")

    pretrained = args.pretrained or ((cfg.get("model", {}) or {}).get("pretrained_path", None))
    if pretrained is None:
        raise ValueError("未提供 --pretrained 且 config 中找不到 model.pretrained_path（可能缺少 pyyaml）")
    if not os.path.isabs(pretrained):
        pretrained = os.path.join(repo_root, "FruVeg26", pretrained) if not pretrained.startswith("FruVeg26") else os.path.join(repo_root, pretrained)
    if not os.path.exists(pretrained):
        raise FileNotFoundError(f"pretrained weights not found: {pretrained}")

    model_path = args.model_path
    if not os.path.isabs(model_path):
        model_path = os.path.join(repo_root, model_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"model_path not found: {model_path}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"device={device} task={args.task}")
    print(f"config={cfg_path}")
    print(f"image_dir={image_dir}")
    print(f"model_path={model_path}")
    print(f"pretrained={pretrained}")

    tfm = _build_transform(cfg)
    image_paths = _list_images(image_dir)
    if len(image_paths) == 0:
        raise RuntimeError(f"no images found in {image_dir}")
    print(f"found {len(image_paths)} images")

    ds = ImagePathDataset(image_paths, tfm)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = _load_model(args.task, model_path=model_path, pretrained_path=pretrained, device=device, cfg=cfg)
    rows = _infer(args.task, model, loader, device)

    out_csv = args.out_csv or os.path.join(image_dir, "pseudo_labels.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    if args.task == "banana" and args.banana_weight_only:
        fieldnames = ["weight"]
    else:
        fieldnames = ["image", "weight", "ripeness", "grade"] if args.task == "banana" else ["image", "weight", "arch", "grade"]
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"saved: {out_csv} (rows={len(rows)})")


if __name__ == "__main__":
    main()

