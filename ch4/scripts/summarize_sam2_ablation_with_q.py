import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def _binary_iou(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    inter = float(np.logical_and(a, b).sum())
    union = float(np.logical_or(a, b).sum())
    return float(inter / (union + 1e-8))


def _largest_cc_ratio(seg: np.ndarray) -> float:
    m = seg.astype(bool)
    total = int(m.sum())
    if total == 0:
        return 0.0
    try:
        from scipy import ndimage  # type: ignore

        labeled, num = ndimage.label(m)
        if num == 0:
            return 0.0
        counts = np.bincount(labeled.ravel())
        counts[0] = 0
        return float(counts.max() / float(total))
    except Exception:
        # 没有 scipy 时给一个保守的退化：单连通性无法可靠计算
        return 1.0


def _read_mask(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path).convert("L"), dtype=np.uint8)
    return (arr > 127).astype(np.uint8)


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="对 SAM2 4项打分消融的输出，用论文 Q 公式重新评价并汇总。")
    p.add_argument(
        "--run_root",
        type=str,
        default=str(ROOT / "results" / "ablation_sam2_masks"),
        help="ablation_sam2_masks 根目录",
    )
    p.add_argument(
        "--dirs",
        nargs="+",
        default=None,
        metavar="NAME",
        help="要汇总的 results 子目录名（如 2026-04-08_01）。不填则自动取 run_root 下所有目录。",
    )
    p.add_argument(
        "--lambda_sam",
        type=float,
        default=0.4,
        help="Q 的 lambda1 (Q_sam 权重)",
    )
    p.add_argument(
        "--lambda_conn",
        type=float,
        default=0.3,
        help="Q 的 lambda2 (Q_conn 权重)",
    )
    p.add_argument(
        "--lambda_smooth",
        type=float,
        default=0.3,
        help="Q 的 lambda3 (Q_smooth 权重)",
    )
    p.add_argument(
        "--out_json",
        type=str,
        default=str(ROOT / "results" / "ablation_sam2_masks" / "q_summary.json"),
        help="汇总 JSON 输出路径",
    )
    return p.parse_args()


def _default_cfg_names() -> list[str]:
    return [
        "full",
        "no_quality",
        "no_center",
        "no_aspect",
        "no_border",
        "only_quality",
        "only_center",
        "only_aspect",
        "only_border",
    ]


def summarize_one_dir(dir_root: Path, cfg_names: list[str], lambdas: tuple[float, float, float]) -> dict[str, Any]:
    lam_sam, lam_conn, lam_smooth = lambdas
    out: dict[str, Any] = {"dir": dir_root.name, "configs": {}}

    for cfg in cfg_names:
        scores_path = dir_root / cfg / "scores.jsonl"
        if not scores_path.exists():
            continue

        rows = list(_iter_jsonl(scores_path))
        # 确保帧顺序稳定：按 image 文件名排序（00.png, 01.png, ...）
        rows.sort(key=lambda r: Path(r["image"]).name)

        q_sam_list: list[float] = []
        q_conn_list: list[float] = []
        masks: list[np.ndarray] = []

        for r in rows:
            parts = r.get("parts", {}) or {}
            pred_iou = float(parts.get("predicted_iou", 0.0))
            stab = float(parts.get("stability_score", 0.0))
            q_sam = 0.5 * (pred_iou + stab)
            q_sam_list.append(float(q_sam))

            mask_path = Path(r["mask"])
            seg = _read_mask(mask_path)
            masks.append(seg)
            q_conn_list.append(float(_largest_cc_ratio(seg)))

        # Q_smooth: 边界帧仅算唯一相邻帧；中间帧算两侧 IoU 的均值
        q_smooth_list: list[float] = []
        T = len(masks)
        for t in range(T):
            if T <= 1:
                q_smooth_list.append(0.0)
            elif t == 0:
                q_smooth_list.append(_binary_iou(masks[0], masks[1]))
            elif t == T - 1:
                q_smooth_list.append(_binary_iou(masks[t], masks[t - 1]))
            else:
                q_smooth_list.append(0.5 * (_binary_iou(masks[t], masks[t - 1]) + _binary_iou(masks[t], masks[t + 1])))

        mean_q_sam = float(np.mean(q_sam_list)) if q_sam_list else 0.0
        mean_q_conn = float(np.mean(q_conn_list)) if q_conn_list else 0.0
        mean_q_smooth = float(np.mean(q_smooth_list)) if q_smooth_list else 0.0
        mean_q = float(lam_sam * mean_q_sam + lam_conn * mean_q_conn + lam_smooth * mean_q_smooth)

        out["configs"][cfg] = {
            "num_frames": int(T),
            "mean_q_sam": mean_q_sam,
            "mean_q_conn": mean_q_conn,
            "mean_q_smooth": mean_q_smooth,
            "mean_q": mean_q,
        }

    return out


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root).expanduser().resolve()
    cfg_names = _default_cfg_names()
    lambdas = (float(args.lambda_sam), float(args.lambda_conn), float(args.lambda_smooth))

    if args.dirs:
        dirs = list(args.dirs)
    else:
        dirs = sorted([p.name for p in run_root.iterdir() if p.is_dir()])

    per_dir = []
    for d in dirs:
        per_dir.append(summarize_one_dir(run_root / d, cfg_names=cfg_names, lambdas=lambdas))

    # 8 目录平均（对每个目录的 mean 再取平均；忽略缺失配置）
    avg: dict[str, dict[str, float]] = {}
    for cfg in cfg_names:
        vals = {"mean_q_sam": [], "mean_q_conn": [], "mean_q_smooth": [], "mean_q": []}
        for d in per_dir:
            if cfg not in d["configs"]:
                continue
            for k in vals:
                vals[k].append(float(d["configs"][cfg][k]))
        if vals["mean_q"]:
            avg[cfg] = {k: float(np.mean(vals[k])) for k in vals}

    out = {
        "run_root": str(run_root),
        "dirs": dirs,
        "lambdas": {"lambda_sam": lambdas[0], "lambda_conn": lambdas[1], "lambda_smooth": lambdas[2]},
        "per_dir": per_dir,
        "avg_over_dirs": avg,
    }

    out_path = Path(args.out_json).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    # 打印表格（TSV 便于直接复制到表格工具）
    print("config\tmean_q_sam\tmean_q_conn\tmean_q_smooth\tmean_q")
    for cfg in cfg_names:
        if cfg not in avg:
            continue
        a = avg[cfg]
        print(f"{cfg}\t{a['mean_q_sam']:.6f}\t{a['mean_q_conn']:.6f}\t{a['mean_q_smooth']:.6f}\t{a['mean_q']:.6f}")
    print(f"WROTE {out_path}")


if __name__ == "__main__":
    main()

