"""
对 morph 序列按 mask.md 中的打分做 mask 选择消融实验。

mask.md 推荐形式：
  Q_t = 0.4 * Q_sam + 0.3 * Q_conn + 0.3 * Q_smooth
  Q_sam = (predicted_iou + stability_score) / 2
  Q_conn = A_max / A（最大连通域占 mask 总面积）
  面积比例 r_t 不在 [r_min, r_max] 时视为低质量（本脚本中从候选集中剔除）

时序项在序列上实现为相邻帧 IoU 之和（与「每对边只计一次」一致，避免对 Q_smooth 双重计数）：
  sum_t F_t(j_t) + w_smooth * sum_t IoU(M_{t-1}, M_t)

运行前请在终端执行: conda activate sam2
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SAM2_DIR = ROOT / "SAM2"
if str(SAM2_DIR) not in sys.path:
    sys.path.insert(0, str(SAM2_DIR))

from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator  # type: ignore  # noqa: E402
from sam2.build_sam import build_sam2  # type: ignore  # noqa: E402


def resolve_device(device: str) -> str:
    """SAM2 mask 默认走 GPU；仅当显式传入 cpu 时才用 CPU。"""
    d = (device or "cuda").strip().lower()
    if d == "cpu":
        return "cpu"
    if d == "auto":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "未检测到 CUDA：本脚本默认在 GPU 上跑 SAM2。请检查 CUDA/驱动，或仅调试时加 --device cpu"
            )
        return "cuda"
    if d.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("指定了 CUDA 设备但 torch.cuda.is_available() 为 False，无法使用 GPU")
        return device.strip()
    return device.strip()


def activate_cuda_device(device: str) -> None:
    if not device.startswith("cuda") or not torch.cuda.is_available():
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


def _largest_cc_ratio(seg: np.ndarray) -> float:
    m = seg.astype(bool)
    total = int(m.sum())
    if total == 0:
        return 0.0
    try:
        from scipy import ndimage

        labeled, num = ndimage.label(m)
        if num == 0:
            return 0.0
        counts = np.bincount(labeled.ravel())
        counts[0] = 0
        return float(counts.max()) / float(total)
    except ImportError:
        pass
    try:
        import cv2

        u8 = m.astype(np.uint8) * 255
        n, _, stats, _ = cv2.connectedComponentsWithStats(u8, connectivity=8)
        if n <= 1:
            return 1.0
        areas = stats[1:, cv2.CC_STAT_AREA]
        return float(areas.max()) / float(total)
    except ImportError:
        return 1.0


def binary_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = float(np.logical_and(a, b).sum())
    union = float(np.logical_or(a, b).sum()) + 1e-6
    return inter / union


def q_smooth_mid(prev_m: np.ndarray, cur_m: np.ndarray, next_m: np.ndarray | None) -> float:
    """与 mask.md 一致：中间帧两边 IoU 平均；首尾只有一侧。"""
    if next_m is None:
        return binary_iou(cur_m, prev_m)
    return 0.5 * (binary_iou(cur_m, prev_m) + binary_iou(cur_m, next_m))


@dataclass(frozen=True)
class MdScoringConfig:
    name: str
    w_sam: float
    w_conn: float
    w_smooth: float
    use_area_gate: bool
    r_min: float
    r_max: float


def _default_experiments() -> list[MdScoringConfig]:
    r_min, r_max = 0.02, 0.85
    return [
        MdScoringConfig("full", 0.4, 0.3, 0.3, True, r_min, r_max),
        MdScoringConfig("no_sam", 0.0, 0.3, 0.3, True, r_min, r_max),
        MdScoringConfig("no_conn", 0.4, 0.0, 0.3, True, r_min, r_max),
        MdScoringConfig("no_smooth", 0.4, 0.3, 0.0, True, r_min, r_max),
        MdScoringConfig("no_area_gate", 0.4, 0.3, 0.3, False, r_min, r_max),
        MdScoringConfig("only_sam", 1.0, 0.0, 0.0, True, r_min, r_max),
        MdScoringConfig("only_conn", 0.0, 1.0, 0.0, True, r_min, r_max),
        MdScoringConfig("only_smooth", 0.0, 0.0, 1.0, True, r_min, r_max),
    ]


def _candidate_features(
    ann: dict[str, Any], h: int, w: int, r_min: float, r_max: float, use_gate: bool
) -> dict[str, Any] | None:
    seg = ann["segmentation"].astype(bool)
    area = float(seg.sum())
    img_area = float(h * w)
    r = area / (img_area + 1e-6)
    if use_gate and (r < r_min or r > r_max):
        return None
    iou_s = float(ann.get("predicted_iou", 0.0))
    stab = float(ann.get("stability_score", 0.0))
    q_sam = 0.5 * (iou_s + stab)
    q_conn = _largest_cc_ratio(seg)
    f_frame = None  # filled per config
    return {
        "seg": seg,
        "area": area,
        "area_ratio": r,
        "q_sam": q_sam,
        "q_conn": q_conn,
        "predicted_iou": iou_s,
        "stability_score": stab,
    }


def _dummy_candidate(h: int, w: int) -> dict[str, Any]:
    return {
        "seg": np.zeros((h, w), dtype=bool),
        "q_sam": 0.0,
        "q_conn": 0.0,
        "area": 0.0,
        "area_ratio": 0.0,
        "predicted_iou": 0.0,
        "stability_score": 0.0,
    }


def _feats_for_config(
    ungated: list[dict[str, Any]], hw: tuple[int, int], cfg: MdScoringConfig
) -> list[dict[str, Any]]:
    h, w = hw
    if not ungated:
        return [_dummy_candidate(h, w)]
    if cfg.use_area_gate:
        g = [f for f in ungated if cfg.r_min <= float(f["area_ratio"]) <= cfg.r_max]
        return g if g else [_dummy_candidate(h, w)]
    return ungated


def _dp_best_path(
    frames_feats: list[list[dict[str, Any]]],
    cfg: MdScoringConfig,
) -> tuple[list[int], float, list[float]]:
    """
    frames_feats[t] 每帧至少一个候选，元素含 'seg', 'q_sam', 'q_conn'。
    优化目标：sum_t (w_sam*q_sam + w_conn*q_conn)_t + w_smooth * sum_t IoU(M_{t-1}, M_t)
    """
    T = len(frames_feats)
    if T == 0:
        return [], 0.0, []

    neg_inf = -1e18
    padded = frames_feats

    def frame_term(ft: dict[str, Any]) -> float:
        return cfg.w_sam * float(ft["q_sam"]) + cfg.w_conn * float(ft["q_conn"])

    dp: list[list[float]] = []
    back: list[list[int]] = []

    n0 = len(padded[0])
    dp0 = [frame_term(padded[0][j]) for j in range(n0)]
    dp.append(dp0)
    back.append([-1] * n0)

    for t in range(1, T):
        nt = len(padded[t])
        nt_prev = len(padded[t - 1])
        cur = [neg_inf] * nt
        cur_back = [-1] * nt
        for j in range(nt):
            seg_j = padded[t][j]["seg"]
            best_val = neg_inf
            best_k = -1
            for k in range(nt_prev):
                seg_k = padded[t - 1][k]["seg"]
                trans = cfg.w_smooth * binary_iou(seg_k, seg_j)
                v = dp[t - 1][k] + trans + frame_term(padded[t][j])
                if v > best_val:
                    best_val = v
                    best_k = k
            cur[j] = best_val
            cur_back[j] = best_k
        dp.append(cur)
        back.append(cur_back)

    last_j = int(np.argmax(dp[-1]))
    total = float(dp[-1][last_j])
    path_idx = [0] * T
    path_idx[T - 1] = last_j
    for t in range(T - 1, 0, -1):
        path_idx[t - 1] = back[t][path_idx[t]]

    smooth_terms: list[float] = []
    masks = [padded[t][path_idx[t]]["seg"] for t in range(T)]
    for t in range(T):
        if T == 1:
            smooth_terms.append(0.0)
        elif t == 0:
            smooth_terms.append(binary_iou(masks[0], masks[1]))
        elif t == T - 1:
            smooth_terms.append(binary_iou(masks[t], masks[t - 1]))
        else:
            smooth_terms.append(q_smooth_mid(masks[t - 1], masks[t], masks[t + 1]))

    return path_idx, total, smooth_terms


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="morph 序列上按 mask.md 公式做 mask 选择消融（DP + 分项分数汇总）。运行前: conda activate sam2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--runs",
        nargs="+",
        default=None,
        metavar="RUN",
        help="批量跑 results/<RUN>/morph（优先级高于 --morph_dirs/--morph_dir；示例：--runs 2026-04-09_01 2026-04-09_02）",
    )
    p.add_argument("--morph_dir", type=str, default="", help="morph 图像目录（默认 results 下最新 */morph）")
    p.add_argument(
        "--morph_dirs",
        nargs="+",
        default=None,
        metavar="DIR",
        help="多个 morph 目录，顺序处理且只加载一次 SAM2（与 --morph_dir 二选一；若同时给出则以此为准）",
    )
    p.add_argument("--results_dir", type=str, default=str(ROOT / "results"))
    p.add_argument("--out_dir", type=str, default=str(ROOT / "results" / "ablation_morph_mask_md"))
    p.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="SAM2 推理设备：默认 cuda（GPU）。调试可加 --device cpu；auto 表示有 GPU 则用 cuda，否则报错",
    )
    p.add_argument("--sam_ckpt", type=str, default="sam2.1_hiera_small.pt")
    p.add_argument("--sam_cfg", type=str, default="configs/sam2.1/sam2.1_hiera_s.yaml")
    p.add_argument("--limit", type=int, default=0, help="只跑前 N 张，0 表示全部")
    p.add_argument("--r_min", type=float, default=0.02)
    p.add_argument("--r_max", type=float, default=0.85)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    activate_cuda_device(device)

    experiments = _default_experiments()
    if args.r_min != 0.02 or args.r_max != 0.85:
        experiments = [
            MdScoringConfig(
                e.name, e.w_sam, e.w_conn, e.w_smooth, e.use_area_gate, float(args.r_min), float(args.r_max)
            )
            for e in experiments
        ]

    results_path = Path(args.results_dir).expanduser().resolve()
    out_base = Path(args.out_dir).expanduser().resolve()

    if args.runs:
        morph_dirs = [(results_path / r / "morph").resolve() for r in args.runs]
    elif args.morph_dirs:
        morph_dirs = [Path(p).expanduser().resolve() for p in args.morph_dirs]
    elif args.morph_dir:
        morph_dirs = [Path(args.morph_dir).expanduser().resolve()]
    else:
        morph_dirs = [_find_latest_morph_dir(results_path)]

    for morph_dir in morph_dirs:
        if not morph_dir.exists():
            raise FileNotFoundError(f"morph_dir 不存在: {morph_dir}")

    mask_generator = build_mask_generator(device=device, sam_ckpt=args.sam_ckpt, sam_cfg=args.sam_cfg)

    overall: dict[str, Any] = {
        "runs": [],
        "aggregate_by_cfg": {},
    }
    agg: dict[str, dict[str, float]] = {}
    agg_frames: dict[str, int] = {}

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

        # 每帧：无面积门控的完整 SAM2 候选（面积门控在 _feats_for_config 里按实验打开）
        ungated_per_frame: list[list[dict[str, Any]]] = []
        hw_per_frame: list[tuple[int, int]] = []
        ref_hw = (256, 256)

        for p_img in imgs:
            try:
                pil = Image.open(p_img).convert("RGB")
            except Exception:
                ungated_per_frame.append([])
                hw_per_frame.append(ref_hw)
                continue
            img_rgb = np.array(pil, dtype=np.uint8, copy=True)
            h, w = img_rgb.shape[:2]
            ref_hw = (h, w)
            hw_per_frame.append((h, w))
            if device.startswith("cuda"):
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    masks = mask_generator.generate(img_rgb)
            else:
                with torch.inference_mode():
                    masks = mask_generator.generate(img_rgb)

            feats: list[dict[str, Any]] = []
            for ann in masks:
                cf = _candidate_features(ann, h, w, 0.0, 1.0, use_gate=False)
                if cf is not None:
                    feats.append(cf)
            ungated_per_frame.append(feats)

        summary: dict[str, Any] = {}

        for cfg in experiments:
            exp_dir = out_root / cfg.name
            exp_dir.mkdir(parents=True, exist_ok=True)

            frames_for_cfg: list[list[dict[str, Any]]] = [
                _feats_for_config(ungated_per_frame[t], hw_per_frame[t], cfg) for t in range(len(imgs))
            ]

            path_idx, total_obj, smooth_report = _dp_best_path(frames_for_cfg, cfg)

            rows: list[dict[str, Any]] = []
            q_sam_list: list[float] = []
            q_conn_list: list[float] = []

            for t, j in enumerate(path_idx):
                p_img = imgs[t]
                if t >= len(frames_for_cfg):
                    continue

                ft = frames_for_cfg[t][j]
                q_sam = float(ft["q_sam"])
                q_conn = float(ft["q_conn"])
                qs = float(smooth_report[t]) if t < len(smooth_report) else 0.0
                md_lin = cfg.w_sam * q_sam + cfg.w_conn * q_conn + cfg.w_smooth * qs
                q_sam_list.append(q_sam)
                q_conn_list.append(q_conn)

                mask = ft["seg"].astype(np.uint8)
                Image.fromarray(mask * 255, mode="L").save(exp_dir / f"{p_img.stem}_mask.png")

                rows.append(
                    {
                        "image": str(p_img),
                        "candidate_idx": int(j),
                        "q_sam": q_sam,
                        "q_conn": q_conn,
                        "q_smooth_md": qs,
                        "md_linear": md_lin,
                        "area_ratio": float(ft["area_ratio"]),
                    }
                )

            with (exp_dir / "scores.jsonl").open("w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

            mean_sam = float(np.mean(q_sam_list)) if q_sam_list else 0.0
            mean_conn = float(np.mean(q_conn_list)) if q_conn_list else 0.0
            mean_smooth = float(np.mean(smooth_report)) if smooth_report else 0.0
            summary[cfg.name] = {
                "dp_objective": float(total_obj),
                "mean_q_sam": mean_sam,
                "mean_q_conn": mean_conn,
                "mean_q_smooth_md": mean_smooth,
                "mean_md_linear_per_frame": float(np.mean([r["md_linear"] for r in rows])) if rows else 0.0,
                "num_frames": len(rows),
            }

        (out_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

        overall["runs"].append(
            {
                "run": morph_dir.parent.name,
                "morph_dir": str(morph_dir),
                "out_root": str(out_root),
                "summary": summary,
            }
        )

        # 聚合：dp_objective 直接求和；均值按帧数加权
        for cfg_name, s in summary.items():
            n = int(s.get("num_frames", 0))
            if cfg_name not in agg:
                agg[cfg_name] = {
                    "dp_objective_sum": 0.0,
                    "mean_q_sam_sum": 0.0,
                    "mean_q_conn_sum": 0.0,
                    "mean_q_smooth_md_sum": 0.0,
                    "mean_md_linear_sum": 0.0,
                }
                agg_frames[cfg_name] = 0
            agg[cfg_name]["dp_objective_sum"] += float(s.get("dp_objective", 0.0))
            agg[cfg_name]["mean_q_sam_sum"] += float(s.get("mean_q_sam", 0.0)) * n
            agg[cfg_name]["mean_q_conn_sum"] += float(s.get("mean_q_conn", 0.0)) * n
            agg[cfg_name]["mean_q_smooth_md_sum"] += float(s.get("mean_q_smooth_md", 0.0)) * n
            agg[cfg_name]["mean_md_linear_sum"] += float(s.get("mean_md_linear_per_frame", 0.0)) * n
            agg_frames[cfg_name] += n

        print("完成（请先 conda activate sam2）。")
        print(f"morph_dir: {morph_dir}")
        print(f"out_root:  {out_root}")
        print("消融分数汇总（见 summary.json）:")
        name_w = max(len(n) for n in summary)
        for name, s in summary.items():
            print(
                f"  {name:<{name_w}}  dp_objective={s['dp_objective']:.4f}  "
                f"mean_q_sam={s['mean_q_sam']:.4f}  mean_q_conn={s['mean_q_conn']:.4f}  "
                f"mean_q_smooth={s['mean_q_smooth_md']:.4f}  mean_md_linear={s['mean_md_linear_per_frame']:.4f}"
            )

    # 总体汇总
    aggregate_by_cfg: dict[str, Any] = {}
    for cfg_name, sums in agg.items():
        n_all = int(agg_frames.get(cfg_name, 0))
        denom = float(n_all) if n_all > 0 else 1.0
        aggregate_by_cfg[cfg_name] = {
            "num_frames_total": n_all,
            "dp_objective_sum": float(sums["dp_objective_sum"]),
            "mean_q_sam": float(sums["mean_q_sam_sum"]) / denom,
            "mean_q_conn": float(sums["mean_q_conn_sum"]) / denom,
            "mean_q_smooth_md": float(sums["mean_q_smooth_md_sum"]) / denom,
            "mean_md_linear_per_frame": float(sums["mean_md_linear_sum"]) / denom,
        }
    overall["aggregate_by_cfg"] = aggregate_by_cfg

    out_base.mkdir(parents=True, exist_ok=True)
    (out_base / "aggregate_summary.json").write_text(
        json.dumps(overall, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n总体汇总（跨全部 runs，见 aggregate_summary.json）:")
    if aggregate_by_cfg:
        name_w = max(len(n) for n in aggregate_by_cfg)
        for name, s in aggregate_by_cfg.items():
            print(
                f"  {name:<{name_w}}  frames={int(s['num_frames_total']):<5d}  "
                f"dp_sum={s['dp_objective_sum']:.4f}  "
                f"mean_q_sam={s['mean_q_sam']:.4f}  mean_q_conn={s['mean_q_conn']:.4f}  "
                f"mean_q_smooth={s['mean_q_smooth_md']:.4f}  mean_md_linear={s['mean_md_linear_per_frame']:.4f}"
            )


if __name__ == "__main__":
    main()
