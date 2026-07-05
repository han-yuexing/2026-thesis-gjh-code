from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageSequence


def _load_frames(path: Path) -> list[Image.Image]:
    img = Image.open(path)
    frames: list[Image.Image] = []

    # PIL 会根据格式不同，n_frames 可能不存在或为 1。
    n_frames = getattr(img, "n_frames", 1)
    if n_frames is None:
        n_frames = 1

    if n_frames > 1:
        for frame in ImageSequence.Iterator(img):
            frames.append(frame.convert("RGB"))
    else:
        frames.append(img.convert("RGB"))

    if not frames:
        raise RuntimeError(f"无法读取任何帧: {path}")
    return frames


def _to_float01(im: Image.Image) -> np.ndarray:
    arr = np.asarray(im, dtype=np.float32) / 255.0
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"期望 RGB 图像, 实际 shape={arr.shape}")
    return arr


def _metrics(a01: np.ndarray, b01: np.ndarray) -> dict[str, float]:
    # a01/b01: (H, W, 3), float32 in [0, 1]
    diff = a01 - b01
    abs_diff = np.abs(diff)
    mae = float(abs_diff.mean())
    mse = float((diff * diff).mean())
    rmse = float(math.sqrt(mse)) if mse >= 0 else float("nan")
    if mse == 0.0:
        psnr = float("inf")
    else:
        # PSNR: 10 * log10(1 / MSE) for normalized [0, 1]
        psnr = float(10.0 * math.log10(1.0 / mse))
    return {"mae": mae, "mse": mse, "rmse": rmse, "psnr": psnr}


def _diff_vis(a01: np.ndarray, b01: np.ndarray) -> Image.Image:
    # 生成一个“差异热力图”：红色越亮表示差异越大。
    # 这里用 per-pixel mean abs difference -> intensity。
    abs_diff = np.abs(a01 - b01)
    intensity = abs_diff.mean(axis=2)  # (H, W) in [0, 1]

    # 轻微拉伸，让小差异也能看见。
    intensity = np.clip(intensity * 2.5, 0.0, 1.0)

    h, w = intensity.shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    vis[..., 0] = (intensity * 255.0).astype(np.uint8)  # R
    vis[..., 1] = (intensity * 170.0).astype(np.uint8)  # G
    vis[..., 2] = (intensity * 40.0).astype(np.uint8)  # B
    # Pillow 新版本会对 mode 参数发弃用警告；vis 本身已是 RGB 数组。
    return Image.fromarray(vis)


def _side_by_side(a: Image.Image, b: Image.Image, diff_vis: Image.Image) -> Image.Image:
    w = a.width
    h = a.height
    if b.width != w or b.height != h or diff_vis.width != w or diff_vis.height != h:
        raise ValueError("三张图尺寸必须一致用于拼接")
    out = Image.new("RGB", (w * 3, h))
    out.paste(a, (0, 0))
    out.paste(b, (w, 0))
    out.paste(diff_vis, (w * 2, 0))
    return out


def _maybe_resize(im: Image.Image, size: tuple[int, int]) -> Image.Image:
    if im.size == size:
        return im
    # 直接 resize（不做保持比例），以保证逐像素比较成立。
    return im.resize(size, resample=Image.BICUBIC)


def compare_images(
    path_a: Path,
    path_b: Path,
    out_dir: Path,
    allow_different_frames: bool,
    resize_to_a: bool,
) -> dict[str, Any]:
    frames_a = _load_frames(path_a)
    frames_b = _load_frames(path_b)

    if not frames_a:
        raise RuntimeError(f"图片A无帧: {path_a}")
    if not frames_b:
        raise RuntimeError(f"图片B无帧: {path_b}")

    n_a = len(frames_a)
    n_b = len(frames_b)

    if n_a != n_b and not allow_different_frames:
        raise RuntimeError(f"帧数不同: A={n_a}, B={n_b}（可加 --allow-different-frames）")

    n = min(n_a, n_b)
    size_a = frames_a[0].size

    out_dir.mkdir(parents=True, exist_ok=True)

    per_frame_metrics: list[dict[str, Any]] = []
    diff_side_paths: list[str] = []

    for i in range(n):
        a = frames_a[i]
        b = frames_b[i]

        if resize_to_a:
            b = _maybe_resize(b, size_a)
            a = _maybe_resize(a, size_a)
        else:
            if a.size != size_a or b.size != size_a:
                raise RuntimeError(
                    f"第 {i} 帧尺寸不一致: A={a.size}, B={b.size}, 参考A={size_a}（可加 --resize-to-a）"
                )

        a01 = _to_float01(a)
        b01 = _to_float01(b)

        m = _metrics(a01, b01)
        diff_vis = _diff_vis(a01, b01)

        side = _side_by_side(a, b, diff_vis)
        side_path = out_dir / f"diff_frame_{i:03d}.png"
        side.save(side_path)

        diff_side_paths.append(str(side_path))
        per_frame_metrics.append({"frame": i, **m})

    # 汇总统计
    def _mean(key: str) -> float:
        vals = [x[key] for x in per_frame_metrics]
        finite = [v for v in vals if math.isfinite(v)]
        if not finite:
            return float("nan")
        return float(sum(finite) / len(finite))

    def _max(key: str) -> float:
        vals = [x[key] for x in per_frame_metrics if math.isfinite(x[key])]
        if not vals:
            return float("nan")
        return float(max(vals))

    summary = {
        "path_a": str(path_a),
        "path_b": str(path_b),
        "frames_compared": n,
        "frames_a": n_a,
        "frames_b": n_b,
        "resize_to_a": resize_to_a,
        "allow_different_frames": allow_different_frames,
        "metric_mean": {
            "mae": _mean("mae"),
            "mse": _mean("mse"),
            "rmse": _mean("rmse"),
            "psnr": _mean("psnr"),
        },
        "metric_max": {
            "mae": _max("mae"),
            "mse": _max("mse"),
            "rmse": _max("rmse"),
            "psnr": _max("psnr"),
        },
    }

    result = {"summary": summary, "per_frame": per_frame_metrics, "diff_side_images": diff_side_paths}
    (out_dir / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="比较两张图像（支持 GIF 逐帧）或比较 morph 文件夹里的逐序号图片。")
    # 文件模式（可选）
    p.add_argument("--a", type=str, default="", help="图片A路径（png/jpg/webp 或 gif）")
    p.add_argument("--b", type=str, default="", help="图片B路径（png/jpg/webp 或 gif）")
    # 目录模式（建议用于对比 run 的 morph 文件夹）
    p.add_argument("--a-morph-dir", type=str, default="", help="A 的 morph 文件夹路径，例如 results/xxx/morph")
    p.add_argument("--b-morph-dir", type=str, default="", help="B 的 morph 文件夹路径，例如 results/yyy/morph")
    p.add_argument(
        "--out-dir",
        default="",
        type=str,
        help="输出目录；不填则自动生成到 results/compare_YYYY-mm-dd_HHMMSS",
    )
    p.add_argument("--allow-different-frames", action="store_true", help="帧数不一致时取较短者对齐比较")
    p.add_argument("--resize-to-a", action="store_true", help="自动把B（以及A的异常帧）resize 到A第一帧尺寸")
    p.add_argument("--exts", type=str, default=".png,.jpg,.jpeg,.webp", help="目录模式下允许的后缀列表")
    p.add_argument("--skip-missing", action="store_true", help="目录模式下跳过缺失的序号（默认会报错）")
    return p


def _list_indexed_images(dir_path: Path, exts: set[str]) -> dict[int, Path]:
    if not dir_path.exists():
        raise FileNotFoundError(f"目录不存在: {dir_path}")
    if not dir_path.is_dir():
        raise NotADirectoryError(f"不是目录: {dir_path}")

    items: dict[int, Path] = {}
    for p in sorted(dir_path.iterdir()):
        if not p.is_file():
            continue
        if p.suffix.lower() not in exts:
            continue

        stem = p.stem
        if not stem.isdigit():
            continue
        idx = int(stem)
        items[idx] = p

    if not items:
        raise RuntimeError(f"目录里未找到可用图片（数字命名如 00.png）: {dir_path}")
    return items


def _compare_single_images(
    path_a: Path,
    path_b: Path,
    out_path: Path,
    resize_to_a: bool,
) -> dict[str, Any]:
    a = Image.open(path_a).convert("RGB")
    b = Image.open(path_b).convert("RGB")

    if resize_to_a:
        b = _maybe_resize(b, a.size)
    else:
        if b.size != a.size:
            raise RuntimeError(f"尺寸不一致: A={a.size}, B={b.size}")

    a01 = _to_float01(a)
    b01 = _to_float01(b)

    m = _metrics(a01, b01)
    diff_vis = _diff_vis(a01, b01)
    side = _side_by_side(a, b, diff_vis)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    side.save(out_path)

    return m


def compare_morph_dirs(
    a_morph_dir: Path,
    b_morph_dir: Path,
    out_dir: Path,
    resize_to_a: bool,
    exts: set[str],
    skip_missing: bool,
) -> dict[str, Any]:
    a_map = _list_indexed_images(a_morph_dir, exts)
    b_map = _list_indexed_images(b_morph_dir, exts)

    a_idxs = set(a_map.keys())
    b_idxs = set(b_map.keys())
    common = sorted(a_idxs & b_idxs)
    missing_a = sorted(b_idxs - a_idxs)
    missing_b = sorted(a_idxs - b_idxs)

    if (missing_a or missing_b) and not skip_missing:
        raise RuntimeError(
            "目录模式下存在缺失序号，使用 --skip-missing 跳过。\n"
            f"缺失（B 中有 A 没有的，按索引）: {missing_a}\n"
            f"缺失（A 中有 B 没有的，按索引）: {missing_b}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)

    per_index: list[dict[str, Any]] = []
    diff_side_paths: list[str] = []

    for idx in common:
        p_a = a_map[idx]
        p_b = b_map[idx]
        side_path = out_dir / f"diff_index_{idx:03d}.png"

        m = _compare_single_images(
            path_a=p_a,
            path_b=p_b,
            out_path=side_path,
            resize_to_a=resize_to_a,
        )
        diff_side_paths.append(str(side_path))
        per_index.append({"index": idx, **m, "path_a": str(p_a), "path_b": str(p_b)})

    def _mean(key: str) -> float:
        vals = [x[key] for x in per_index]
        finite = [v for v in vals if math.isfinite(v)]
        if not finite:
            return float("nan")
        return float(sum(finite) / len(finite))

    def _max(key: str) -> float:
        vals = [x[key] for x in per_index if math.isfinite(x[key])]
        if not vals:
            return float("nan")
        return float(max(vals))

    summary = {
        "mode": "morph_dir",
        "path_a_dir": str(a_morph_dir),
        "path_b_dir": str(b_morph_dir),
        "indices_compared": len(common),
        "indices_a_total": len(a_map),
        "indices_b_total": len(b_map),
        "resize_to_a": resize_to_a,
        "skip_missing": skip_missing,
        "missing_in_a": missing_a,
        "missing_in_b": missing_b,
        "metric_mean": {
            "mae": _mean("mae"),
            "mse": _mean("mse"),
            "rmse": _mean("rmse"),
            "psnr": _mean("psnr"),
        },
        "metric_max": {
            "mae": _max("mae"),
            "mse": _max("mse"),
            "rmse": _max("rmse"),
            "psnr": _max("psnr"),
        },
    }

    result = {"summary": summary, "per_index": per_index, "diff_side_images": diff_side_paths}
    (out_dir / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    args = _build_argparser().parse_args()

    if args.out_dir:
        out_dir = Path(args.out_dir).expanduser().resolve()
    else:
        ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        out_dir = (Path(__file__).resolve().parent / "results" / f"compare_{ts}").resolve()

    if args.a_morph_dir and args.b_morph_dir:
        exts = {s.strip().lower() for s in args.exts.split(",") if s.strip()}
        res = compare_morph_dirs(
            a_morph_dir=Path(args.a_morph_dir).expanduser().resolve(),
            b_morph_dir=Path(args.b_morph_dir).expanduser().resolve(),
            out_dir=out_dir,
            resize_to_a=bool(args.resize_to_a),
            exts=exts,
            skip_missing=bool(args.skip_missing),
        )
        mmean = res["summary"]["metric_mean"]
        print(f"输出目录: {out_dir}")
        print(
            "平均指标: "
            f"MAE={mmean['mae']:.6f}, MSE={mmean['mse']:.6f}, RMSE={mmean['rmse']:.6f}, "
            f"PSNR={mmean['psnr'] if math.isfinite(mmean['psnr']) else 'inf'}"
        )
        return

    # 文件模式
    if not args.a or not args.b:
        raise RuntimeError("请提供 --a/--b（文件模式）或 --a-morph-dir/--b-morph-dir（目录模式）。")

    path_a = Path(args.a).expanduser().resolve()
    path_b = Path(args.b).expanduser().resolve()
    if not path_a.exists():
        raise FileNotFoundError(f"--a 不存在: {path_a}")
    if not path_b.exists():
        raise FileNotFoundError(f"--b 不存在: {path_b}")

    res = compare_images(
        path_a=path_a,
        path_b=path_b,
        out_dir=out_dir,
        allow_different_frames=bool(args.allow_different_frames),
        resize_to_a=bool(args.resize_to_a),
    )
    # 尽量在终端打印关键指标，方便你直接看结果。
    mmean = res["summary"]["metric_mean"]
    print(f"输出目录: {out_dir}")
    print(
        "平均指标: "
        f"MAE={mmean['mae']:.6f}, MSE={mmean['mse']:.6f}, RMSE={mmean['rmse']:.6f}, "
        f"PSNR={mmean['psnr'] if math.isfinite(mmean['psnr']) else 'inf'}"
    )


if __name__ == "__main__":
    main()

