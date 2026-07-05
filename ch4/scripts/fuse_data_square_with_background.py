import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def iter_images(root: Path) -> Iterable[Path]:
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def mask_path_for_image(img_path: Path, img_root: Path, mask_root: Path) -> Path:
    rel = img_path.relative_to(img_root)
    stem_no_ext = rel.with_suffix("")  # e.g. train/1
    return (mask_root / stem_no_ext.parent / f"{stem_no_ext.name}_mask.png").resolve()


def load_backgrounds(bg_600: Path, bg_3024: Path) -> dict[tuple[int, int], Image.Image]:
    b600 = Image.open(bg_600).convert("RGB")
    b3024 = Image.open(bg_3024).convert("RGB")
    return {(600, 600): b600, (3024, 3024): b3024}


def choose_background_by_category(
    backgrounds: dict[tuple[int, int], Image.Image],
    *,
    banana_bg: Image.Image | None,
    rel_path: Path,
    fallback: str,
) -> Image.Image:
    """
    规则：
    - 黄瓜（cucumber_*）一律用 600 背景
    - 香蕉（banana_*）一律用 3024 背景
    其他类别兜底策略与旧版一致（nearest / 3024）。
    """
    parts = {p.lower() for p in rel_path.parts}
    if any("cucumber" in p for p in parts):
        return backgrounds[(600, 600)]
    if any("banana" in p for p in parts):
        return banana_bg if banana_bg is not None else backgrounds[(3024, 3024)]

    if fallback == "nearest":
        # 没有类别信息时，按训练图画布尺寸去近似（由调用者保证传入的背景会 resize 到目标尺寸）
        return backgrounds[(3024, 3024)] if (3024, 3024) in backgrounds else next(iter(backgrounds.values()))
    return backgrounds[(3024, 3024)]


def fuse_one(img_path: Path, mask_path: Path, bg: Image.Image) -> Image.Image:
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    bg_resized = bg.resize((w, h), resample=Image.BICUBIC)

    m = Image.open(mask_path).convert("L")
    if m.size != (w, h):
        m = m.resize((w, h), resample=Image.NEAREST)

    img_np = np.asarray(img, dtype=np.float32)
    bg_np = np.asarray(bg_resized, dtype=np.float32)
    m_np = (np.asarray(m, dtype=np.float32) / 255.0)[:, :, None]  # (H,W,1)

    out = img_np * m_np + bg_np * (1.0 - m_np)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return Image.fromarray(out)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="将 data_square 图像按 mask 与指定背景融合。")
    p.add_argument("--img_dir", type=str, default=str(ROOT / "data_square"))
    p.add_argument("--mask_dir", type=str, default=str(ROOT / "data_square_sam2_masks"))
    p.add_argument("--out_dir", type=str, default=str(ROOT / "data_square_fused"))
    p.add_argument("--bg_600", type=str, default=str(ROOT / "data_square" / "back_600.png"))
    p.add_argument("--bg_3024", type=str, default=str(ROOT / "data_square" / "back_3024.png"))
    p.add_argument("--bg_banana", type=str, default=str(ROOT / "back1.png"), help="香蕉融合用背景（默认 back1.png）")
    p.add_argument("--only_banana", action="store_true", help="只处理香蕉图像（黄瓜/其他类别不覆盖输出）")
    p.add_argument(
        "--fallback",
        type=str,
        default="nearest",
        choices=["nearest", "3024"],
        help="当图片尺寸不是 600/3024 时，选择背景策略：nearest=选最接近尺寸并缩放；3024=永远用3024并缩放",
    )
    p.add_argument("--limit", type=int, default=0, help="只处理前 N 张（0 表示全部）")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    img_dir = Path(args.img_dir).expanduser().resolve()
    mask_dir = Path(args.mask_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    backgrounds = load_backgrounds(Path(args.bg_600), Path(args.bg_3024))
    banana_bg_path = Path(args.bg_banana).expanduser().resolve()
    banana_bg = Image.open(banana_bg_path).convert("RGB") if banana_bg_path.exists() else None

    imgs = list(iter_images(img_dir))
    # 排除背景图本身，避免被当作训练图处理
    bg_names = {Path(args.bg_600).name, Path(args.bg_3024).name}
    imgs = [p for p in imgs if p.name not in bg_names]
    if args.only_banana:
        imgs = [p for p in imgs if "banana" in str(p.relative_to(img_dir)).lower()]
    imgs.sort()
    if args.limit and args.limit > 0:
        imgs = imgs[: int(args.limit)]
    if not imgs:
        raise RuntimeError(f"{img_dir} 下未找到任何图片")

    missing_masks = 0
    for idx, p_img in enumerate(imgs, start=1):
        p_mask = mask_path_for_image(p_img, img_root=img_dir, mask_root=mask_dir)
        if not p_mask.exists():
            missing_masks += 1
            continue

        rel = p_img.relative_to(img_dir)
        bg = choose_background_by_category(backgrounds, banana_bg=banana_bg, rel_path=rel, fallback=args.fallback)
        fused = fuse_one(p_img, p_mask, bg)

        out_path = (out_dir / rel).with_suffix(p_img.suffix.lower())
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fused.save(out_path)

        if idx == 1 or idx % 50 == 0 or idx == len(imgs):
            print(f"[{idx}/{len(imgs)}] {rel} -> {out_path.relative_to(out_dir)}")

    if missing_masks:
        print(f"警告：缺少 mask 的图片数 = {missing_masks}（已跳过）")
    print("完成。")
    print(f"img_dir:  {img_dir}")
    print(f"mask_dir: {mask_dir}")
    print(f"out_dir:  {out_dir}")


if __name__ == "__main__":
    main()

