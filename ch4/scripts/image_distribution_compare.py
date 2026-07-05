import argparse
import csv
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageFile

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _HAS_MPL = True
except Exception:
    matplotlib = None  # type: ignore
    plt = None  # type: ignore
    _HAS_MPL = False

try:
    from scipy import ndimage as _ndimage  # type: ignore
except Exception:
    _ndimage = None


ImageFile.LOAD_TRUNCATED_IMAGES = True


_IMG_EXTS_DEFAULT = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")


@dataclass(frozen=True)
class ImageStats:
    path: str
    group: str
    width: int
    height: int
    aspect: float
    mean_r: float
    mean_g: float
    mean_b: float
    std_r: float
    std_g: float
    std_b: float
    y_mean: float
    y_std: float
    sat_mean: float
    sat_std: float
    edge_density: float


def _iter_image_paths(root: Path, exts: tuple[str, ...]) -> list[Path]:
    if not root.exists():
        return []
    paths: list[Path] = []
    for dp, _, fns in os.walk(str(root)):
        for fn in fns:
            p = Path(dp) / fn
            if p.suffix.lower() in exts:
                paths.append(p)
    return sorted(paths)


def _rgb01(im: Image.Image) -> np.ndarray:
    arr = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"期望 RGB 图像, 实际 shape={arr.shape}")
    return arr


def _rgb_to_y(rgb01: np.ndarray) -> np.ndarray:
    # ITU-R BT.601 luma
    return 0.299 * rgb01[..., 0] + 0.587 * rgb01[..., 1] + 0.114 * rgb01[..., 2]


def _rgb_to_hsv_sat(rgb01: np.ndarray) -> np.ndarray:
    # 返回 S 通道（[0,1]），向量化实现，避免逐像素 colorsys 慢
    r = rgb01[..., 0]
    g = rgb01[..., 1]
    b = rgb01[..., 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    diff = mx - mn
    # HSV: S = diff / mx (mx==0 -> 0)
    sat = np.zeros_like(mx)
    nonzero = mx > 1e-8
    sat[nonzero] = diff[nonzero] / mx[nonzero]
    return sat


def _edge_density(gray01: np.ndarray) -> float:
    # 轻量“边缘密度”：Sobel 梯度幅值 > 阈值 的像素比例
    if _ndimage is None:
        return float("nan")
    gx = _ndimage.sobel(gray01, axis=1, mode="reflect")
    gy = _ndimage.sobel(gray01, axis=0, mode="reflect")
    mag = np.sqrt(gx * gx + gy * gy)
    thr = float(np.quantile(mag, 0.9))  # 自适应阈值，稳一些
    if not math.isfinite(thr) or thr <= 0:
        return 0.0
    return float((mag > thr).mean())


def _safe_float(x: float) -> float:
    if not math.isfinite(x):
        return float("nan")
    return float(x)


def compute_stats(path: Path, group: str) -> ImageStats:
    im = Image.open(path)
    w, h = im.size
    rgb01 = _rgb01(im)

    mean = rgb01.reshape(-1, 3).mean(axis=0)
    std = rgb01.reshape(-1, 3).std(axis=0)

    y = _rgb_to_y(rgb01)
    sat = _rgb_to_hsv_sat(rgb01)

    return ImageStats(
        path=str(path),
        group=group,
        width=int(w),
        height=int(h),
        aspect=_safe_float(float(w) / float(h) if h > 0 else float("nan")),
        mean_r=_safe_float(float(mean[0])),
        mean_g=_safe_float(float(mean[1])),
        mean_b=_safe_float(float(mean[2])),
        std_r=_safe_float(float(std[0])),
        std_g=_safe_float(float(std[1])),
        std_b=_safe_float(float(std[2])),
        y_mean=_safe_float(float(y.mean())),
        y_std=_safe_float(float(y.std())),
        sat_mean=_safe_float(float(sat.mean())),
        sat_std=_safe_float(float(sat.std())),
        edge_density=_safe_float(_edge_density(y)),
    )


def _describe(vals: np.ndarray) -> dict[str, float]:
    vals = vals.astype(np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan"), "p05": float("nan"), "p50": float("nan"), "p95": float("nan")}
    return {
        "n": int(vals.size),
        "mean": float(vals.mean()),
        "std": float(vals.std()),
        "p05": float(np.quantile(vals, 0.05)),
        "p50": float(np.quantile(vals, 0.50)),
        "p95": float(np.quantile(vals, 0.95)),
    }


def _plot_overlay_hist(
    out_path: Path,
    a: np.ndarray,
    b: np.ndarray,
    a_name: str,
    b_name: str,
    title: str,
    bins: int,
    range_: tuple[float, float] | None = None,
) -> None:
    if not _HAS_MPL:
        return
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    plt.figure(figsize=(7.5, 5.2), dpi=160)
    plt.hist(a, bins=bins, range=range_, density=True, alpha=0.45, label=a_name, color="#1f77b4")
    plt.hist(b, bins=bins, range=range_, density=True, alpha=0.45, label=b_name, color="#ff7f0e")
    plt.title(title)
    plt.legend(loc="best")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def _plot_size_scatter(out_path: Path, stats: list[ImageStats], gen_name: str, real_name: str) -> None:
    if not _HAS_MPL:
        return
    ws_g, hs_g, ws_r, hs_r = [], [], [], []
    for s in stats:
        if s.group == "gen":
            ws_g.append(s.width)
            hs_g.append(s.height)
        else:
            ws_r.append(s.width)
            hs_r.append(s.height)

    plt.figure(figsize=(7.5, 5.2), dpi=160)
    if ws_r:
        plt.scatter(ws_r, hs_r, s=10, alpha=0.45, label=real_name, color="#ff7f0e")
    if ws_g:
        plt.scatter(ws_g, hs_g, s=10, alpha=0.45, label=gen_name, color="#1f77b4")
    plt.xlabel("width")
    plt.ylabel("height")
    plt.title("分辨率散点图")
    plt.legend(loc="best")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def _write_csv(out_path: Path, stats: list[ImageStats]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(stats[0]).keys()))
        w.writeheader()
        for s in stats:
            w.writerow(asdict(s))


def _subsample(paths: list[Path], max_n: int, seed: int) -> list[Path]:
    if max_n <= 0 or len(paths) <= max_n:
        return paths
    rng = random.Random(seed)
    idxs = list(range(len(paths)))
    rng.shuffle(idxs)
    idxs = sorted(idxs[:max_n])
    return [paths[i] for i in idxs]


def _to_np(stats: list[ImageStats], key: str, group: str) -> np.ndarray:
    vals: list[float] = []
    for s in stats:
        if s.group != group:
            continue
        vals.append(float(getattr(s, key)))
    return np.asarray(vals, dtype=np.float32)


def main() -> None:
    p = argparse.ArgumentParser(description="对比生成图像 vs 原始图像的分布（尺寸/颜色/亮度/饱和度/边缘等）。")
    p.add_argument("--gen_dir", required=True, type=str, help="生成图像目录（递归扫描）")
    p.add_argument("--real_dir", required=True, type=str, help="原始图像目录（递归扫描）")
    p.add_argument("--out_dir", default="", type=str, help="输出目录（默认写到 results/imgdist_时间戳）")
    p.add_argument("--max_images", default=2000, type=int, help="每个目录最多抽样多少张（<=0 表示全量）")
    p.add_argument("--seed", default=0, type=int, help="抽样随机种子")
    p.add_argument("--exts", default=",".join(_IMG_EXTS_DEFAULT), type=str, help="允许的图片后缀，逗号分隔")
    args = p.parse_args()

    gen_dir = Path(args.gen_dir).expanduser().resolve()
    real_dir = Path(args.real_dir).expanduser().resolve()
    exts = tuple(s.strip().lower() for s in args.exts.split(",") if s.strip())

    if args.out_dir:
        out_dir = Path(args.out_dir).expanduser().resolve()
    else:
        ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        out_dir = (Path(__file__).resolve().parents[1] / "results" / f"imgdist_{ts}").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    gen_paths = _iter_image_paths(gen_dir, exts)
    real_paths = _iter_image_paths(real_dir, exts)
    if not gen_paths:
        raise RuntimeError(f"未在生成目录找到图片: {gen_dir}")
    if not real_paths:
        raise RuntimeError(f"未在原始目录找到图片: {real_dir}")

    gen_paths = _subsample(gen_paths, int(args.max_images), int(args.seed))
    real_paths = _subsample(real_paths, int(args.max_images), int(args.seed) + 1)

    stats: list[ImageStats] = []
    bad: list[dict[str, str]] = []

    def _run(paths: Iterable[Path], group: str) -> None:
        for pp in paths:
            try:
                stats.append(compute_stats(pp, group=group))
            except Exception as e:
                bad.append({"path": str(pp), "group": group, "error": repr(e)})

    _run(gen_paths, "gen")
    _run(real_paths, "real")
    if not stats:
        raise RuntimeError("没有任何图片成功读取/统计。")

    # 输出 per-image 表
    _write_csv(out_dir / "per_image.csv", stats)
    (out_dir / "bad_images.json").write_text(json.dumps(bad, ensure_ascii=False, indent=2), encoding="utf-8")

    # 汇总统计 + 画图
    gen_name = f"gen ({len([s for s in stats if s.group == 'gen'])})"
    real_name = f"real ({len([s for s in stats if s.group == 'real'])})"

    keys = [
        ("aspect", "长宽比分布", 60, None),
        ("y_mean", "亮度均值 (Y) 分布", 60, (0.0, 1.0)),
        ("y_std", "亮度标准差 (Y) 分布", 60, (0.0, 0.5)),
        ("sat_mean", "饱和度均值 (S) 分布", 60, (0.0, 1.0)),
        ("sat_std", "饱和度标准差 (S) 分布", 60, (0.0, 0.5)),
        ("std_r", "R 通道标准差分布", 60, (0.0, 0.5)),
        ("std_g", "G 通道标准差分布", 60, (0.0, 0.5)),
        ("std_b", "B 通道标准差分布", 60, (0.0, 0.5)),
        ("edge_density", "边缘密度分布（Sobel, 若无 scipy 则为 NaN）", 60, (0.0, 1.0)),
    ]

    summary: dict[str, object] = {
        "gen_dir": str(gen_dir),
        "real_dir": str(real_dir),
        "out_dir": str(out_dir),
        "exts": list(exts),
        "max_images": int(args.max_images),
        "seed": int(args.seed),
        "count_ok": {"gen": int(sum(1 for s in stats if s.group == "gen")), "real": int(sum(1 for s in stats if s.group == "real"))},
        "count_bad": {"gen": int(sum(1 for x in bad if x["group"] == "gen")), "real": int(sum(1 for x in bad if x["group"] == "real"))},
        "has_matplotlib": bool(_HAS_MPL),
        "has_scipy": bool(_ndimage is not None),
        "features": {},
    }

    feat_summary: dict[str, dict[str, dict[str, float]]] = {}
    for k, title, bins, rg in keys:
        a = _to_np(stats, k, "gen")
        b = _to_np(stats, k, "real")
        feat_summary[k] = {"gen": _describe(a), "real": _describe(b)}
        _plot_overlay_hist(out_dir / "plots" / f"{k}.png", a, b, gen_name, real_name, title, bins=bins, range_=rg)
    summary["features"] = feat_summary

    # 尺寸散点图 + 宽/高直方图
    _plot_size_scatter(out_dir / "plots" / "size_scatter.png", stats, gen_name=gen_name, real_name=real_name)
    _plot_overlay_hist(
        out_dir / "plots" / "width.png",
        _to_np(stats, "width", "gen"),
        _to_np(stats, "width", "real"),
        gen_name,
        real_name,
        "宽度分布",
        bins=60,
        range_=None,
    )
    _plot_overlay_hist(
        out_dir / "plots" / "height.png",
        _to_np(stats, "height", "gen"),
        _to_np(stats, "height", "real"),
        gen_name,
        real_name,
        "高度分布",
        bins=60,
        range_=None,
    )

    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("输出目录:", out_dir)


if __name__ == "__main__":
    main()

