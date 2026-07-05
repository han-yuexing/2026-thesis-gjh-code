import argparse
import os
import shutil
import contextlib
from datetime import date
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image


ROOT = Path(__file__).resolve().parent
DIFFMORPHER_DIR = ROOT / "diffmorpher"
SAM2_DIR = ROOT / "SAM2"

# Make local packages importable.
import sys

if str(DIFFMORPHER_DIR) not in sys.path:
    sys.path.insert(0, str(DIFFMORPHER_DIR))
if str(SAM2_DIR) not in sys.path:
    sys.path.insert(0, str(SAM2_DIR))

from model import DiffMorpherPipeline  # type: ignore  # noqa: E402
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
    # cuda:N
    idx_str = device.split(":", 1)[1]
    idx = int(idx_str)
    if idx < 0 or idx >= torch.cuda.device_count():
        raise ValueError(f"无效 CUDA 设备索引: {device}, 当前可用卡数={torch.cuda.device_count()}")
    torch.cuda.set_device(idx)


def pick_input_images(assets_dir: Path, image0: str = "", image1: str = "") -> List[Path]:
    if image0 and image1:
        p0 = assets_dir / image0
        p1 = assets_dir / image1
        if not p0.exists() or not p1.exists():
            raise FileNotFoundError(f"指定图片不存在: {p0} / {p1}")
        return [p0, p1]

    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    imgs = sorted([p for p in assets_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])
    if len(imgs) < 2:
        raise RuntimeError(f"{assets_dir} 中可用图片少于 2 张")
    return imgs[:2]


def ensure_dirs(results_root: Path) -> tuple[Path, Path, Path]:
    today = date.today().strftime("%Y-%m-%d")
    idx = 1
    while True:
        day_dir = results_root / f"{today}_{idx:02d}"
        if not day_dir.exists():
            break
        idx += 1
    morph_dir = day_dir / "morph"
    mask_dir = day_dir / "mask"
    morph_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    return day_dir, morph_dir, mask_dir


def run_diffmorpher(
    img0: Path,
    img1: Path,
    morph_dir: Path,
    device: str,
    num_frames: int,
    prompt0: str,
    prompt1: str,
    use_lora: bool,
    save_lora_dir: str,
) -> None:
    activate_cuda_device(device)
    ckpt_path = DIFFMORPHER_DIR / "v2-1_512-ema-pruned.ckpt"
    cfg_path = DIFFMORPHER_DIR / "v2-inference.yaml"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"未找到 DiffMorpher 权重: {ckpt_path}")
    if not cfg_path.exists():
        raise FileNotFoundError(f"未找到 SD 配置: {cfg_path}")
    # 关键：与 diffmorpher/main.py 一致，在 diffmorpher 目录下加载。
    # 这样 "stabilityai/stable-diffusion-2" 这类相对路径会命中本地目录，而不是去 HuggingFace。
    old_cwd = Path.cwd()
    with contextlib.ExitStack():
        os.chdir(DIFFMORPHER_DIR)
        try:
            sd_pipe = StableDiffusionPipeline.from_single_file(
                "v2-1_512-ema-pruned.ckpt",
                original_config_file="v2-inference.yaml",
                torch_dtype=torch.float32,
                local_files_only=True,
            )
            pipeline = DiffMorpherPipeline(
                vae=sd_pipe.vae,
                text_encoder=sd_pipe.text_encoder,
                tokenizer=sd_pipe.tokenizer,
                unet=sd_pipe.unet,
                scheduler=sd_pipe.scheduler,
                safety_checker=getattr(sd_pipe, "safety_checker", None),
                feature_extractor=getattr(sd_pipe, "feature_extractor", None),
            )
            pipeline.to(device)
            _ = pipeline(
                img_path_0=str(img0),
                img_path_1=str(img1),
                prompt_0=prompt0,
                prompt_1=prompt1,
                save_lora_dir=save_lora_dir,
                output_path=str(morph_dir),
                num_frames=num_frames,
                save_intermediates=True,
                use_lora=use_lora,
                use_adain=True,
                use_reschedule=True,
                guidance_scale=1.0,
                device=device,
            )
        finally:
            os.chdir(old_cwd)


def build_mask_generator(device: str, sam_ckpt: str, sam_cfg: str) -> SAM2AutomaticMaskGenerator:
    ckpt = SAM2_DIR / "checkpoints" / sam_ckpt
    model_cfg = sam_cfg
    if not ckpt.exists():
        raise FileNotFoundError(f"未找到 SAM2 权重: {ckpt}")

    sam2_model = build_sam2(model_cfg, str(ckpt), device=device)
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


def select_best_mask(
    masks: list,
    image_shape: tuple[int, int, int],
    prev_mask: np.ndarray | None = None,
):
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
        center_dist = np.sqrt((cx - cx_img) ** 2 + (cy - cy_img) ** 2)
        center_score = -center_dist / max(h, w)

        iou_score = float(ann.get("predicted_iou", 0.0))
        stability = float(ann.get("stability_score", 0.0))
        area_score = area / img_area

        border_margin = 5
        touch_border = (
            x <= border_margin
            or y <= border_margin
            or x + bw >= w - border_margin
            or y + bh >= h - border_margin
        )
        border_penalty = -0.2 if touch_border else 0.0

        # 强抑制背景大块区域（常见于桌面/墙面/上半屏）。
        huge_area = area_score > 0.35
        full_width = (bw / max(w, 1)) > 0.95
        very_tall = (bh / max(h, 1)) > 0.80
        bg_like_penalty = -2.0 if (huge_area or (full_width and touch_border) or (touch_border and very_tall)) else 0.0

        # 时序一致性：更偏向与上一帧面积接近、位置接近、重叠更高的候选。
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

        # 通用主体打分：质量 + 面积 + 居中，避免只适配细长物体。
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

    # 兜底：若都被过滤，仍避免选到覆盖全屏的大背景。
    filtered = []
    for ann in masks:
        x, y, bw, bh = ann.get("bbox", [0, 0, 0, 0])
        area_ratio = float(ann.get("area", 0)) / img_area
        if area_ratio <= 0.60 and not (bw >= 0.98 * w and bh >= 0.98 * h):
            filtered.append(ann)
    if filtered:
        return max(filtered, key=lambda x: x.get("predicted_iou", 0.0))
    return max(masks, key=lambda x: x.get("predicted_iou", 0.0))


def run_sam2_masks(morph_dir: Path, mask_dir: Path, device: str, sam_ckpt: str, sam_cfg: str) -> None:
    activate_cuda_device(device)
    mask_generator = build_mask_generator(device, sam_ckpt=sam_ckpt, sam_cfg=sam_cfg)
    morph_imgs = sorted(morph_dir.glob("*.png"))
    if not morph_imgs:
        raise RuntimeError(f"{morph_dir} 中没有可分割的 morph 图像")

    prev_mask = None
    for p in morph_imgs:
        img_bgr = cv2.imread(str(p))
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        if device.startswith("cuda"):
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                masks = mask_generator.generate(img_rgb)
        else:
            with torch.inference_mode():
                masks = mask_generator.generate(img_rgb)

        best = select_best_mask(masks, img_rgb.shape, prev_mask=prev_mask)
        if best is None:
            mask = np.zeros(img_rgb.shape[:2], dtype=np.uint8)
        else:
            mask = best["segmentation"].astype(np.uint8)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        prev_mask = mask.copy()

        out_path = mask_dir / f"{p.stem}_mask.png"
        cv2.imwrite(str(out_path), mask * 255)


def save_gif_from_images(image_paths: list[Path], out_gif: Path, duration_ms: int = 100) -> None:
    if not image_paths:
        return
    frames = [Image.open(p).convert("RGB") for p in image_paths]
    frames[0].save(
        str(out_gif),
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="DiffMorpher + SAM2 pipeline")
    parser.add_argument("--assets_dir", type=str, default=str(ROOT / "assets"))
    parser.add_argument("--results_dir", type=str, default=str(ROOT / "results"))
    parser.add_argument("--image0", type=str, default="", help="assets 内第一张图名")
    parser.add_argument("--image1", type=str, default="", help="assets 内第二张图名")
    parser.add_argument("--prompt0", type=str, default="A photo of an object")
    parser.add_argument("--prompt1", type=str, default="A photo of an object")
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--device", type=str, default="auto", help="全局设备: auto/cpu/cuda/cuda:0")
    parser.add_argument("--diff_device", type=str, default="", help="DiffMorpher 单独设备，留空则跟随 --device")
    parser.add_argument("--sam_device", type=str, default="", help="SAM2 单独设备，留空则跟随 --device")
    parser.add_argument("--no_lora", action="store_true", help="关闭 DiffMorpher 的 LoRA（默认开启）")
    # 留空则自动设置为 results/今日日期/lora
    parser.add_argument("--save_lora_dir", type=str, default="")
    # 与 SAM2/test.py 一致的默认模型：small + sam2.1_hiera_s.yaml
    parser.add_argument("--sam_ckpt", type=str, default="sam2.1_hiera_small.pt")
    parser.add_argument("--sam_cfg", type=str, default="configs/sam2.1/sam2.1_hiera_s.yaml")
    parser.add_argument("--gif_duration", type=int, default=100, help="输出 GIF 帧间隔（毫秒）")
    return parser.parse_args()


def main():
    args = parse_args()
    base_device = resolve_device(args.device)
    diff_device = resolve_device(args.diff_device) if args.diff_device else base_device
    sam_device = resolve_device(args.sam_device) if args.sam_device else base_device

    assets_dir = Path(args.assets_dir)
    results_dir = Path(args.results_dir)
    img0, img1 = pick_input_images(assets_dir, args.image0, args.image1)
    day_dir, morph_dir, mask_dir = ensure_dirs(results_dir)

    save_lora_dir = args.save_lora_dir.strip() if isinstance(args.save_lora_dir, str) else ""
    if not save_lora_dir:
        save_lora_dir = str(day_dir / "lora")

    # 清理旧的同名输出，避免历史残留干扰本次结果。
    for old_png in morph_dir.glob("*.png"):
        old_png.unlink(missing_ok=True)
    for old_png in mask_dir.glob("*.png"):
        old_png.unlink(missing_ok=True)
    runlog = morph_dir / "run.log"
    if runlog.exists():
        runlog.unlink()

    print(f"[1/2] DiffMorpher: {img0.name} + {img1.name} -> {morph_dir}")
    print(f"DiffMorpher device: {diff_device}")
    print(f"DiffMorpher use_lora: {not args.no_lora}")
    if diff_device.startswith("cuda") and torch.cuda.is_available():
        activate_cuda_device(diff_device)
        print(f"DiffMorpher current cuda device: cuda:{torch.cuda.current_device()}")
    run_diffmorpher(
        img0=img0,
        img1=img1,
        morph_dir=morph_dir,
        device=diff_device,
        num_frames=args.num_frames,
        prompt0=args.prompt0,
        prompt1=args.prompt1,
        use_lora=not args.no_lora,
        save_lora_dir=save_lora_dir,
    )

    # run.log 保留一份到日期目录，便于整体查看。
    if runlog.exists():
        shutil.copy2(runlog, morph_dir.parent / "diffmorpher_run.log")

    print(f"[2/2] SAM2 mask: {morph_dir} -> {mask_dir}")
    print(f"SAM2 device: {sam_device}")
    if sam_device.startswith("cuda") and torch.cuda.is_available():
        activate_cuda_device(sam_device)
        print(f"SAM2 current cuda device: cuda:{torch.cuda.current_device()}")
    run_sam2_masks(
        morph_dir=morph_dir,
        mask_dir=mask_dir,
        device=sam_device,
        sam_ckpt=args.sam_ckpt,
        sam_cfg=args.sam_cfg,
    )

    morph_pngs = sorted(morph_dir.glob("*.png"))
    mask_pngs = sorted(mask_dir.glob("*_mask.png"))
    morph_gif = day_dir / "morph.gif"
    mask_gif = day_dir / "mask.gif"
    save_gif_from_images(morph_pngs, morph_gif, duration_ms=args.gif_duration)
    save_gif_from_images(mask_pngs, mask_gif, duration_ms=args.gif_duration)

    print("完成。")
    print(f"Morph 输出: {morph_dir}")
    print(f"Mask 输出:  {mask_dir}")
    print(f"Morph GIF: {morph_gif}")
    print(f"Mask GIF:  {mask_gif}")


if __name__ == "__main__":
    main()
