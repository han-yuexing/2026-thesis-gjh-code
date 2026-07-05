import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SAM2_DIR = ROOT / "SAM2"

# Make SAM2 local package importable.
import sys

if str(SAM2_DIR) not in sys.path:
    sys.path.insert(0, str(SAM2_DIR))

from sam2.build_sam import build_sam2  # type: ignore  # noqa: E402
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator  # type: ignore  # noqa: E402


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return device


def activate_cuda_device(device: str) -> None:
    if not device.startswith("cuda"):
        return
    if not torch.cuda.is_available():
        return
    if device == "cuda":
        torch.cuda.set_device(0)
        return
    idx = int(device.split(":", 1)[1])
    torch.cuda.set_device(idx)


def build_mask_generator(device: str, sam_ckpt: str, sam_cfg: str) -> SAM2AutomaticMaskGenerator:
    ckpt = SAM2_DIR / "checkpoints" / sam_ckpt
    if not ckpt.exists():
        raise FileNotFoundError(f"未找到 SAM2 权重: {ckpt}")
    sam2_model = build_sam2(sam_cfg, str(ckpt), device=device)
    return SAM2AutomaticMaskGenerator(
        model=sam2_model,
        points_per_side=32,
        points_per_batch=64,
        pred_iou_thresh=0.7,
        stability_score_thresh=0.92,
        stability_score_offset=0.7,
        crop_n_layers=1,
        box_nms_thresh=0.7,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=200,
        use_m2m=True,
    )


def _iter_morph_images(morph_dir: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    imgs = sorted([p for p in morph_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])
    if not imgs:
        raise RuntimeError(f"{morph_dir} 中没有找到任何图片")
    return imgs


def _find_latest_morph_dir(results_dir: Path) -> Path:
    candidates: list[Path] = []
    for p in results_dir.iterdir():
        if not p.is_dir():
            continue
        morph = p / "morph"
        if morph.is_dir():
            candidates.append(morph)
    if not candidates:
        raise RuntimeError(f"在 {results_dir} 下未找到任何形如 results/*/morph 的目录")
    candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return candidates[0]


@dataclass(frozen=True)
class ScoringConfig:
    name: str
    w_quality: float
    w_center: float
    w_aspect: float
    enable_border_penalty: bool = True
    border_penalty_value: float = -0.3
    border_margin: int = 5
    min_area_ratio: float = 0.01


def _score_ann(ann: dict[str, Any], image_shape: tuple[int, int, int], cfg: ScoringConfig) -> tuple[float, dict[str, float]]:
    h, w = image_shape[:2]
    img_area = float(h * w)

    area = float(ann.get("area", 0.0))
    x, y, bw, bh = ann.get("bbox", [0, 0, 0, 0])
    x = float(x)
    y = float(y)
    bw = float(bw)
    bh = float(bh)

    cx_img, cy_img = w / 2.0, h / 2.0
    cx = x + bw / 2.0
    cy = y + bh / 2.0

    center_dist = float(np.sqrt((cx - cx_img) ** 2 + (cy - cy_img) ** 2))
    center_score = -center_dist / float(max(h, w))

    aspect = max(bw / (bh + 1e-6), bh / (bw + 1e-6))
    aspect_score = float(min(aspect, 8.0) / 8.0)

    iou_score = float(ann.get("predicted_iou", 0.0))
    stability_score = float(ann.get("stability_score", 0.0))
    quality_score = 0.6 * iou_score + 0.4 * stability_score

    touch_border = (
        x <= cfg.border_margin
        or y <= cfg.border_margin
        or x + bw >= w - cfg.border_margin
        or y + bh >= h - cfg.border_margin
    )
    border_penalty = cfg.border_penalty_value if (cfg.enable_border_penalty and touch_border) else 0.0

    score = cfg.w_quality * quality_score + cfg.w_center * center_score + cfg.w_aspect * aspect_score + border_penalty
    parts = {
        "quality_score": float(quality_score),
        "center_score": float(center_score),
        "aspect_score": float(aspect_score),
        "border_penalty": float(border_penalty),
        "predicted_iou": float(iou_score),
        "stability_score": float(stability_score),
        "area": float(area),
        "area_ratio": float(area / (img_area + 1e-6)),
    }
    return float(score), parts


def pick_best_mask_for_cfg(masks: list[dict[str, Any]], image_shape: tuple[int, int, int], cfg: ScoringConfig):
    h, w = image_shape[:2]
    img_area = float(h * w)
    best = None
    best_score = -1e18
    best_parts: dict[str, float] | None = None

    for ann in masks:
        area = float(ann.get("area", 0.0))
        if area < cfg.min_area_ratio * img_area:
            continue
        score, parts = _score_ann(ann, image_shape=image_shape, cfg=cfg)
        if score > best_score:
            best = ann
            best_score = score
            best_parts = parts

    return best, float(best_score), best_parts


def _default_experiments() -> list[ScoringConfig]:
    # 对应 SAM2/test.py (57-61) 的四项消融：quality/center/aspect/border_penalty
    return [
        ScoringConfig(name="full", w_quality=1.2, w_center=0.8, w_aspect=0.8, enable_border_penalty=True),
        ScoringConfig(name="no_quality", w_quality=0.0, w_center=0.8, w_aspect=0.8, enable_border_penalty=True),
        ScoringConfig(name="no_center", w_quality=1.2, w_center=0.0, w_aspect=0.8, enable_border_penalty=True),
        ScoringConfig(name="no_aspect", w_quality=1.2, w_center=0.8, w_aspect=0.0, enable_border_penalty=True),
        ScoringConfig(name="no_border", w_quality=1.2, w_center=0.8, w_aspect=0.8, enable_border_penalty=False),
        ScoringConfig(name="only_quality", w_quality=1.2, w_center=0.0, w_aspect=0.0, enable_border_penalty=False),
        ScoringConfig(name="only_center", w_quality=0.0, w_center=0.8, w_aspect=0.0, enable_border_penalty=False),
        ScoringConfig(name="only_aspect", w_quality=0.0, w_center=0.0, w_aspect=0.8, enable_border_penalty=False),
        ScoringConfig(name="only_border", w_quality=0.0, w_center=0.0, w_aspect=0.0, enable_border_penalty=True),
    ]


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="对 SAM2/test.py 的 mask 选择打分做消融实验：既保存每组 best mask，也导出每张图所有候选 mask 在各配置下的打分。"
    )
    p.add_argument("--morph_dir", type=str, default="", help="输入 morph 图像目录（默认自动选 results 下最新的 */morph）")
    p.add_argument(
        "--morph_dirs",
        nargs="+",
        default=None,
        metavar="DIR",
        help="多个 morph 目录，顺序处理且只加载一次 SAM2（与 --morph_dir 二选一；若同时给出则以此为准）",
    )
    p.add_argument("--results_dir", type=str, default=str(ROOT / "results"), help="用于自动查找最新 morph 的 results 根目录")
    p.add_argument("--out_dir", type=str, default=str(ROOT / "results" / "ablation_sam2_masks"), help="消融输出根目录")
    p.add_argument("--device", type=str, default="auto", help="auto/cpu/cuda/cuda:0")
    p.add_argument("--sam_ckpt", type=str, default="sam2.1_hiera_small.pt")
    p.add_argument("--sam_cfg", type=str, default="configs/sam2.1/sam2.1_hiera_s.yaml")
    p.add_argument("--limit", type=int, default=0, help="只跑前 N 张（0 表示全跑）")
    p.add_argument("--no_postprocess", action="store_true", help="不做形态学后处理（默认本脚本不依赖 opencv，仍会跳过后处理）")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    activate_cuda_device(device)

    results_path = Path(args.results_dir).expanduser().resolve()
    out_base = Path(args.out_dir).expanduser().resolve()

    if args.morph_dirs:
        morph_dirs = [Path(p).expanduser().resolve() for p in args.morph_dirs]
    elif args.morph_dir:
        morph_dirs = [Path(args.morph_dir).expanduser().resolve()]
    else:
        morph_dirs = [_find_latest_morph_dir(results_path)]

    for morph_dir in morph_dirs:
        if not morph_dir.exists():
            raise FileNotFoundError(f"morph_dir 不存在: {morph_dir}")

    experiments = _default_experiments()
    mask_generator = build_mask_generator(device=device, sam_ckpt=args.sam_ckpt, sam_cfg=args.sam_cfg)

    for morph_dir in morph_dirs:
        out_root = out_base / morph_dir.parent.name
        out_root.mkdir(parents=True, exist_ok=True)

        (out_root / "experiments.json").write_text(
            json.dumps([asdict(e) for e in experiments], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        imgs = _iter_morph_images(morph_dir)
        if args.limit and args.limit > 0:
            imgs = imgs[: int(args.limit)]

        exp_rows: dict[str, list[dict[str, Any]]] = {e.name: [] for e in experiments}

        for p_img in imgs:
            try:
                pil = Image.open(p_img).convert("RGB")
            except Exception:
                continue
            # 避免 torchvision 对 non-writable numpy 给出 warning
            img_rgb = np.array(pil, dtype=np.uint8, copy=True)

            if device.startswith("cuda"):
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    masks = mask_generator.generate(img_rgb)
            else:
                with torch.inference_mode():
                    masks = mask_generator.generate(img_rgb)

            # 逐候选 mask：对每个候选在所有消融配置下计算 score/分项，导出到单独文件。
            candidates_dir = out_root / "candidates_scores"
            candidates_dir.mkdir(parents=True, exist_ok=True)
            per_candidate_rows: list[dict[str, Any]] = []
            for cand_idx, ann in enumerate(masks):
                bbox = ann.get("bbox", [0, 0, 0, 0])
                area = float(ann.get("area", 0.0))
                predicted_iou = float(ann.get("predicted_iou", 0.0))
                stability_score = float(ann.get("stability_score", 0.0))

                scores_by_cfg: dict[str, Any] = {}
                for cfg in experiments:
                    score, parts = _score_ann(ann, image_shape=img_rgb.shape, cfg=cfg)
                    scores_by_cfg[cfg.name] = {"score": float(score), "parts": parts}

                per_candidate_rows.append(
                    {
                        "image": str(p_img),
                        "candidate_idx": int(cand_idx),
                        "bbox": bbox,
                        "area": float(area),
                        "predicted_iou": float(predicted_iou),
                        "stability_score": float(stability_score),
                        "scores": scores_by_cfg,
                    }
                )

            _write_jsonl(candidates_dir / f"{p_img.stem}.jsonl", per_candidate_rows)

            for cfg in experiments:
                best, best_score, best_parts = pick_best_mask_for_cfg(masks, img_rgb.shape, cfg)
                if best is None:
                    mask = np.zeros(img_rgb.shape[:2], dtype=np.uint8)
                    bbox = [0, 0, 0, 0]
                    area = 0.0
                    parts = {}
                else:
                    mask = best["segmentation"].astype(np.uint8)
                    bbox = best.get("bbox", [0, 0, 0, 0])
                    area = float(best.get("area", 0.0))
                    parts = best_parts or {}

                # 这里不强依赖 opencv：默认直接保存 raw mask。
                # 如确实需要与 test.py 完全一致的形态学处理，请在有 opencv 的环境里跑 pipeline.py 的 run_sam2_masks。

                exp_dir = out_root / cfg.name
                exp_dir.mkdir(parents=True, exist_ok=True)
                out_mask_path = exp_dir / f"{p_img.stem}_mask.png"
                Image.fromarray((mask * 255).astype(np.uint8), mode="L").save(out_mask_path)

                exp_rows[cfg.name].append(
                    {
                        "image": str(p_img),
                        "mask": str(out_mask_path),
                        "best_score": float(best_score),
                        "bbox": bbox,
                        "area": float(area),
                        "parts": parts,
                    }
                )

        for cfg in experiments:
            _write_jsonl(out_root / cfg.name / "scores.jsonl", exp_rows[cfg.name])

        print("完成。")
        print(f"morph_dir: {morph_dir}")
        print(f"out_root:  {out_root}")


if __name__ == "__main__":
    main()
