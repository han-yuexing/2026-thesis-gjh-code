import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SAM2_DIR = ROOT / "SAM2"

# Make SAM2 local package importable.
import sys

if str(SAM2_DIR) not in sys.path:
    sys.path.insert(0, str(SAM2_DIR))

from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator  # type: ignore  # noqa: E402
from sam2.build_sam import build_sam2  # type: ignore  # noqa: E402


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
    idx_str = device.split(":", 1)[1]
    idx = int(idx_str)
    torch.cuda.set_device(idx)


def build_mask_generator(
    device: str,
    sam_ckpt: str,
    sam_cfg: str,
    *,
    points_per_side: int,
    points_per_batch: int,
    pred_iou_thresh: float,
    stability_score_thresh: float,
    stability_score_offset: float,
    crop_n_layers: int,
    box_nms_thresh: float,
    crop_n_points_downscale_factor: int,
    min_mask_region_area: int,
    use_m2m: bool,
) -> SAM2AutomaticMaskGenerator:
    ckpt = SAM2_DIR / "checkpoints" / sam_ckpt
    if not ckpt.exists():
        raise FileNotFoundError(f"未找到 SAM2 权重: {ckpt}")
    sam2_model = build_sam2(sam_cfg, str(ckpt), device=device)
    return SAM2AutomaticMaskGenerator(
        model=sam2_model,
        points_per_side=int(points_per_side),
        points_per_batch=int(points_per_batch),
        pred_iou_thresh=float(pred_iou_thresh),
        stability_score_thresh=float(stability_score_thresh),
        stability_score_offset=float(stability_score_offset),
        crop_n_layers=int(crop_n_layers),
        box_nms_thresh=float(box_nms_thresh),
        crop_n_points_downscale_factor=int(crop_n_points_downscale_factor),
        min_mask_region_area=int(min_mask_region_area),
        use_m2m=bool(use_m2m),
    )


def select_best_mask(masks: list, image_shape: tuple[int, int, int], prev_mask: np.ndarray | None = None):
    """
    复用 pipeline.py 的主体选择逻辑（偏向主体、抑制大背景，并可用上一帧做轻微时序一致性）。
    对 data_square 这种单帧图像，prev_mask 默认不启用。
    """
    if not masks:
        return None
    h, w = image_shape[:2]
    img_area = h * w
    cx_img, cy_img = w / 2.0, h / 2.0
    best = None
    best_score = -1e18
    for ann in masks:
        area = float(ann.get("area", 0))
        if area < 0.01 * img_area:
            continue

        x, y, bw, bh = ann.get("bbox", [0, 0, 0, 0])
        cx = x + bw / 2.0
        cy = y + bh / 2.0
        center_dist = float(np.sqrt((cx - cx_img) ** 2 + (cy - cy_img) ** 2))
        center_score = -center_dist / float(max(h, w))

        iou_score = float(ann.get("predicted_iou", 0.0))
        stability = float(ann.get("stability_score", 0.0))
        area_score = float(area / float(img_area))

        border_margin = 5
        touch_border = (
            x <= border_margin
            or y <= border_margin
            or x + bw >= w - border_margin
            or y + bh >= h - border_margin
        )
        border_penalty = -0.2 if touch_border else 0.0

        huge_area = area_score > 0.35
        full_width = (bw / max(w, 1)) > 0.95
        very_tall = (bh / max(h, 1)) > 0.80
        bg_like_penalty = -2.0 if (huge_area or (full_width and touch_border) or (touch_border and very_tall)) else 0.0

        temporal_score = 0.0
        if prev_mask is not None:
            cur = ann["segmentation"].astype(np.uint8)
            prev = (prev_mask > 0).astype(np.uint8)
            inter = float(np.logical_and(cur, prev).sum())
            union = float(np.logical_or(cur, prev).sum()) + 1e-6
            iou_prev = inter / union
            prev_area = float(prev.sum()) + 1e-6
            area_ratio_diff = abs((area / prev_area) - 1.0)
            temporal_score = 1.5 * iou_prev - 0.5 * min(area_ratio_diff, 2.0)

        score = (
            1.0 * iou_score
            + 0.8 * stability
            + 0.6 * area_score
            + 0.5 * center_score
            + border_penalty
            + bg_like_penalty
            + temporal_score
        )
        if score > best_score:
            best = ann
            best_score = score
    if best is not None:
        return best

    filtered = []
    for ann in masks:
        x, y, bw, bh = ann.get("bbox", [0, 0, 0, 0])
        area_ratio = float(ann.get("area", 0)) / float(img_area)
        if area_ratio <= 0.60 and not (bw >= 0.98 * w and bh >= 0.98 * h):
            filtered.append(ann)
    if filtered:
        return max(filtered, key=lambda x: x.get("predicted_iou", 0.0))
    return max(masks, key=lambda x: x.get("predicted_iou", 0.0))


def iter_images(root: Path) -> Iterable[Path]:
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in exts:
            continue
        yield p


def maybe_morphology(mask01: np.ndarray, enable: bool) -> np.ndarray:
    if not enable:
        return mask01
    try:
        import cv2  # type: ignore
    except Exception:
        return mask01
    kernel = np.ones((5, 5), np.uint8)
    out = cv2.morphologyEx(mask01.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    out = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel)
    return out.astype(np.uint8)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="对 data_square 下每张图像用 SAM2 自动提取 best mask。")
    p.add_argument("--in_dir", type=str, default=str(ROOT / "data_square"), help="输入目录（递归扫描图片）")
    p.add_argument(
        "--out_dir",
        type=str,
        default=str(ROOT / "results" / "data_square_sam2_masks"),
        help="输出目录（保持相对路径结构，保存 *_mask.png）",
    )
    p.add_argument("--device", type=str, default="auto", help="auto/cpu/cuda/cuda:0")
    p.add_argument("--sam_ckpt", type=str, default="sam2.1_hiera_small.pt")
    p.add_argument("--sam_cfg", type=str, default="configs/sam2.1/sam2.1_hiera_s.yaml")
    p.add_argument("--limit", type=int, default=0, help="只处理前 N 张（0 表示全部）")
    p.add_argument("--morphology", action="store_true", help="对 mask 做 close+open 后处理（需要 opencv）")
    # 显存/速度调参：默认值与 pipeline.py 保持一致，但允许你在显存紧张时降低开销。
    p.add_argument("--points_per_side", type=int, default=32)
    p.add_argument("--points_per_batch", type=int, default=64)
    p.add_argument("--crop_n_layers", type=int, default=1)
    p.add_argument("--pred_iou_thresh", type=float, default=0.7)
    p.add_argument("--stability_score_thresh", type=float, default=0.92)
    p.add_argument("--stability_score_offset", type=float, default=0.7)
    p.add_argument("--box_nms_thresh", type=float, default=0.7)
    p.add_argument("--crop_n_points_downscale_factor", type=int, default=2)
    p.add_argument("--min_mask_region_area", type=int, default=200)
    p.add_argument("--no_m2m", action="store_true", help="关闭 m2m refine（显著省显存，通常够用）")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    in_dir = Path(args.in_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    activate_cuda_device(device)

    mask_generator = build_mask_generator(
        device=device,
        sam_ckpt=args.sam_ckpt,
        sam_cfg=args.sam_cfg,
        points_per_side=args.points_per_side,
        points_per_batch=args.points_per_batch,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
        stability_score_offset=args.stability_score_offset,
        crop_n_layers=args.crop_n_layers,
        box_nms_thresh=args.box_nms_thresh,
        crop_n_points_downscale_factor=args.crop_n_points_downscale_factor,
        min_mask_region_area=args.min_mask_region_area,
        use_m2m=(not bool(args.no_m2m)),
    )

    imgs = list(iter_images(in_dir))
    imgs.sort()
    if args.limit and args.limit > 0:
        imgs = imgs[: int(args.limit)]
    if not imgs:
        raise RuntimeError(f"{in_dir} 下未找到任何图片")

    for idx, p_img in enumerate(imgs, start=1):
        try:
            pil = Image.open(p_img).convert("RGB")
        except Exception:
            continue
        img_rgb = np.array(pil, dtype=np.uint8, copy=True)

        if device.startswith("cuda"):
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                masks = mask_generator.generate(img_rgb)
        else:
            with torch.inference_mode():
                masks = mask_generator.generate(img_rgb)

        best = select_best_mask(masks, img_rgb.shape, prev_mask=None)
        if best is None:
            mask01 = np.zeros(img_rgb.shape[:2], dtype=np.uint8)
        else:
            mask01 = best["segmentation"].astype(np.uint8)
        mask01 = maybe_morphology(mask01, enable=bool(args.morphology))

        rel = p_img.relative_to(in_dir)
        out_path = (out_dir / rel).with_suffix("")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_mask = out_path.parent / f"{out_path.name}_mask.png"
        Image.fromarray((mask01 * 255).astype(np.uint8), mode="L").save(out_mask)

        if idx % 50 == 0 or idx == 1 or idx == len(imgs):
            print(f"[{idx}/{len(imgs)}] {rel} -> {out_mask.relative_to(out_dir)}")

    print("完成。")
    print(f"in_dir:  {in_dir}")
    print(f"out_dir: {out_dir}")


if __name__ == "__main__":
    main()

