import argparse
import csv
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise RuntimeError(f"CSV 无表头: {path}")
        rows: list[dict[str, str]] = []
        for r in reader:
            rows.append({k: (v if v is not None else "") for k, v in r.items()})
        return list(reader.fieldnames), rows


def _to_float(x: str) -> float:
    x = (x or "").strip()
    if x == "":
        return float("nan")
    try:
        return float(x)
    except Exception:
        return float("nan")


def _infer_numeric_columns(rows: list[dict[str, str]], cols: list[str], min_valid_ratio: float) -> list[str]:
    out: list[str] = []
    n = len(rows)
    if n == 0:
        return out
    for c in cols:
        vals = np.asarray([_to_float(r.get(c, "")) for r in rows], dtype=np.float64)
        valid = np.isfinite(vals).sum()
        if valid / max(1, n) >= float(min_valid_ratio) and valid >= 5:
            out.append(c)
    return out


def _infer_label_column(cols: list[str]) -> str | None:
    cand = ["label", "level", "grade", "class", "y"]
    m = {c.lower(): c for c in cols}
    for k in cand:
        if k in m:
            return m[k]
    return None


def _overlay_hist(
    out_path: Path,
    a: np.ndarray,
    b: np.ndarray,
    a_name: str,
    b_name: str,
    title: str,
    bins: int = 60,
) -> None:
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 and b.size == 0:
        return

    # 统一范围，避免一边被截断
    lo = float(np.nanmin([a.min() if a.size else np.nan, b.min() if b.size else np.nan]))
    hi = float(np.nanmax([a.max() if a.size else np.nan, b.max() if b.size else np.nan]))
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        return

    plt.figure(figsize=(7.5, 5.2), dpi=160)
    plt.hist(a, bins=bins, range=(lo, hi), density=True, alpha=0.45, label=a_name, color="#1f77b4")
    plt.hist(b, bins=bins, range=(lo, hi), density=True, alpha=0.45, label=b_name, color="#ff7f0e")
    plt.title(title)
    plt.legend(loc="best")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def _bar_compare(
    out_path: Path,
    a_cnt: Counter[str],
    b_cnt: Counter[str],
    a_name: str,
    b_name: str,
    title: str,
) -> None:
    keys = sorted(set(a_cnt.keys()) | set(b_cnt.keys()))
    if not keys:
        return
    a_vals = [a_cnt.get(k, 0) for k in keys]
    b_vals = [b_cnt.get(k, 0) for k in keys]

    x = np.arange(len(keys))
    w = 0.38

    plt.figure(figsize=(7.5, 5.2), dpi=160)
    plt.bar(x - w / 2, a_vals, width=w, label=a_name, color="#1f77b4", alpha=0.8)
    plt.bar(x + w / 2, b_vals, width=w, label=b_name, color="#ff7f0e", alpha=0.8)
    plt.xticks(x, keys, rotation=25, ha="right")
    plt.title(title)
    plt.legend(loc="best")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def main() -> None:
    p = argparse.ArgumentParser(description="画两个 CSV 的分布对比图（数值列直方图 + 标签柱状图）。")
    p.add_argument("--a", required=True, type=str, help="CSV A 路径")
    p.add_argument("--b", required=True, type=str, help="CSV B 路径")
    p.add_argument("--a_name", default="A", type=str)
    p.add_argument("--b_name", default="B", type=str)
    p.add_argument("--min_valid_ratio", default=0.9, type=float, help="判定为数值列所需的有效数值比例")
    p.add_argument("--numeric_cols", default="", type=str, help="指定数值列（逗号分隔），不填则自动推断")
    p.add_argument("--label_col_a", default="", type=str, help="A 的标签列名（不填则自动推断）")
    p.add_argument("--label_col_b", default="", type=str, help="B 的标签列名（不填则自动推断）")
    p.add_argument("--out_dir", default="", type=str, help="输出目录（默认 results/csvdist_时间戳）")
    args = p.parse_args()

    a_path = Path(args.a).expanduser().resolve()
    b_path = Path(args.b).expanduser().resolve()
    if not a_path.exists():
        raise FileNotFoundError(f"A 不存在: {a_path}")
    if not b_path.exists():
        raise FileNotFoundError(f"B 不存在: {b_path}")

    a_cols, a_rows = _read_csv(a_path)
    b_cols, b_rows = _read_csv(b_path)
    if not a_rows or not b_rows:
        raise RuntimeError("CSV 为空，无法画分布。")

    if args.out_dir.strip():
        out_dir = Path(args.out_dir).expanduser().resolve()
    else:
        ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        out_dir = (Path(__file__).resolve().parents[1] / "results" / f"csvdist_{ts}").resolve()
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 数值列
    if args.numeric_cols.strip():
        numeric = [x.strip() for x in args.numeric_cols.split(",") if x.strip()]
    else:
        a_num = set(_infer_numeric_columns(a_rows, a_cols, float(args.min_valid_ratio)))
        b_num = set(_infer_numeric_columns(b_rows, b_cols, float(args.min_valid_ratio)))
        numeric = sorted(a_num & b_num)

    # 标签列（类别分布）
    label_a = args.label_col_a.strip() or (_infer_label_column(a_cols) or "")
    label_b = args.label_col_b.strip() or (_infer_label_column(b_cols) or "")

    # 画数值列直方图
    numeric_done: list[str] = []
    for c in numeric:
        a_vals = np.asarray([_to_float(r.get(c, "")) for r in a_rows], dtype=np.float64)
        b_vals = np.asarray([_to_float(r.get(c, "")) for r in b_rows], dtype=np.float64)
        _overlay_hist(plots_dir / f"{c}.png", a_vals, b_vals, args.a_name, args.b_name, f"{c} 分布对比")
        numeric_done.append(c)

    # 画标签柱状图（如果能推断到）
    label_done = False
    if label_a and label_b and (label_a in a_cols) and (label_b in b_cols):
        a_cnt = Counter((r.get(label_a, "") or "").strip() for r in a_rows)
        b_cnt = Counter((r.get(label_b, "") or "").strip() for r in b_rows)
        _bar_compare(plots_dir / "label_dist.png", a_cnt, b_cnt, args.a_name, args.b_name, f"{label_a} vs {label_b} 标签分布")
        label_done = True

    summary: dict[str, Any] = {
        "a_path": str(a_path),
        "b_path": str(b_path),
        "a_name": args.a_name,
        "b_name": args.b_name,
        "count_rows": {"a": len(a_rows), "b": len(b_rows)},
        "numeric_cols_compared": numeric_done,
        "label_col_a": label_a,
        "label_col_b": label_b,
        "label_plot_written": bool(label_done),
        "out_dir": str(out_dir),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("输出目录:", out_dir)
    print("数值列对比:", numeric_done if numeric_done else "(无)")
    print("标签列:", label_a, "vs", label_b, "label_plot=", label_done)


if __name__ == "__main__":
    main()

