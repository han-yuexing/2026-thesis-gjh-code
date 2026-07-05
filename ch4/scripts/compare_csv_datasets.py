import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RowKey:
    key: str


def _read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise RuntimeError(f"CSV 无表头: {path}")
        rows: list[dict[str, str]] = []
        for r in reader:
            rows.append({k: (v if v is not None else "") for k, v in r.items()})
        return rows


def _pick_key_field(fieldnames: list[str]) -> str:
    # 优先 image / path / file 之类
    cand = ["image", "img", "path", "filepath", "file", "filename", "name"]
    lowered = {f.lower(): f for f in fieldnames}
    for c in cand:
        if c in lowered:
            return lowered[c]
    # 退化：取第一列
    return fieldnames[0]


def _normalize_key(v: str, normalize_basename: bool) -> str:
    v = (v or "").strip()
    if normalize_basename:
        try:
            return Path(v).name
        except Exception:
            return v
    return v


def _count_dups(keys: list[str]) -> dict[str, int]:
    c = Counter(keys)
    return {k: n for k, n in c.items() if n > 1}


def _infer_label_field(fieldnames: list[str]) -> str | None:
    cand = ["label", "level", "grade", "class", "y"]
    lowered = {f.lower(): f for f in fieldnames}
    for c in cand:
        if c in lowered:
            return lowered[c]
    return None


def main() -> None:
    p = argparse.ArgumentParser(description="对比两个 CSV 数据集（样本集合/重复/标签分布/冲突等）。")
    p.add_argument("--a", required=True, type=str, help="CSV A 路径（例如 banana_masked.csv 或 pseudo_labels.csv）")
    p.add_argument("--b", required=True, type=str, help="CSV B 路径（例如 train.csv）")
    p.add_argument("--a_name", default="A", type=str, help="A 的显示名")
    p.add_argument("--b_name", default="B", type=str, help="B 的显示名")
    p.add_argument("--key_field_a", default="", type=str, help="A 用哪一列作为主键（默认自动选择）")
    p.add_argument("--key_field_b", default="", type=str, help="B 用哪一列作为主键（默认自动选择）")
    p.add_argument("--label_field_a", default="", type=str, help="A 用哪一列作为标签列（默认自动推断）")
    p.add_argument("--label_field_b", default="", type=str, help="B 用哪一列作为标签列（默认自动推断）")
    p.add_argument("--normalize_basename", action="store_true", help="主键统一用 basename（路径转文件名）")
    p.add_argument("--out_dir", default="", type=str, help="输出目录（默认 results/csvdiff_时间戳）")
    args = p.parse_args()

    a_path = Path(args.a).expanduser().resolve()
    b_path = Path(args.b).expanduser().resolve()
    if not a_path.exists():
        raise FileNotFoundError(f"A 不存在: {a_path}")
    if not b_path.exists():
        raise FileNotFoundError(f"B 不存在: {b_path}")

    a_rows = _read_csv(a_path)
    b_rows = _read_csv(b_path)
    if not a_rows:
        raise RuntimeError(f"A 为空: {a_path}")
    if not b_rows:
        raise RuntimeError(f"B 为空: {b_path}")

    a_fields = list(a_rows[0].keys())
    b_fields = list(b_rows[0].keys())

    key_a = args.key_field_a.strip() or _pick_key_field(a_fields)
    key_b = args.key_field_b.strip() or _pick_key_field(b_fields)
    if key_a not in a_fields:
        raise RuntimeError(f"A key_field 不存在: {key_a}，可选: {a_fields}")
    if key_b not in b_fields:
        raise RuntimeError(f"B key_field 不存在: {key_b}，可选: {b_fields}")

    label_a = args.label_field_a.strip() or (_infer_label_field(a_fields) or "")
    label_b = args.label_field_b.strip() or (_infer_label_field(b_fields) or "")
    if label_a and label_a not in a_fields:
        raise RuntimeError(f"A label_field 不存在: {label_a}，可选: {a_fields}")
    if label_b and label_b not in b_fields:
        raise RuntimeError(f"B label_field 不存在: {label_b}，可选: {b_fields}")

    a_keys = [_normalize_key(r.get(key_a, ""), bool(args.normalize_basename)) for r in a_rows]
    b_keys = [_normalize_key(r.get(key_b, ""), bool(args.normalize_basename)) for r in b_rows]

    a_dups = _count_dups(a_keys)
    b_dups = _count_dups(b_keys)

    a_set = set(a_keys)
    b_set = set(b_keys)
    inter = sorted(a_set & b_set)
    only_a = sorted(a_set - b_set)
    only_b = sorted(b_set - a_set)

    # 标签分布（如果有）
    def _label_counter(rows: list[dict[str, str]], key_field: str, label_field: str, normalize: bool) -> Counter[str]:
        c: Counter[str] = Counter()
        if not label_field:
            return c
        for r in rows:
            k = _normalize_key(r.get(key_field, ""), normalize)
            y = (r.get(label_field, "") or "").strip()
            if not k:
                continue
            c[y] += 1
        return c

    a_label_cnt = _label_counter(a_rows, key_a, label_a, bool(args.normalize_basename))
    b_label_cnt = _label_counter(b_rows, key_b, label_b, bool(args.normalize_basename))

    # 同名样本标签冲突（如果两边都有标签）
    conflicts: list[dict[str, Any]] = []
    if label_a and label_b and inter:
        a_map: dict[str, str] = {}
        for r in a_rows:
            k = _normalize_key(r.get(key_a, ""), bool(args.normalize_basename))
            if not k or k in a_map:
                continue
            a_map[k] = (r.get(label_a, "") or "").strip()

        b_map: dict[str, str] = {}
        for r in b_rows:
            k = _normalize_key(r.get(key_b, ""), bool(args.normalize_basename))
            if not k or k in b_map:
                continue
            b_map[k] = (r.get(label_b, "") or "").strip()

        for k in inter:
            ya = a_map.get(k, "")
            yb = b_map.get(k, "")
            if ya != yb:
                conflicts.append({"key": k, "a_label": ya, "b_label": yb})

    # 输出
    if args.out_dir.strip():
        out_dir = Path(args.out_dir).expanduser().resolve()
    else:
        ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        out_dir = (Path(__file__).resolve().parents[1] / "results" / f"csvdiff_{ts}").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "a_path": str(a_path),
        "b_path": str(b_path),
        "a_name": args.a_name,
        "b_name": args.b_name,
        "key_field_a": key_a,
        "key_field_b": key_b,
        "label_field_a": label_a,
        "label_field_b": label_b,
        "normalize_basename": bool(args.normalize_basename),
        "count_rows": {"a": len(a_rows), "b": len(b_rows)},
        "count_unique_keys": {"a": len(a_set), "b": len(b_set)},
        "count_intersection": len(inter),
        "count_only_a": len(only_a),
        "count_only_b": len(only_b),
        "count_dups": {"a": len(a_dups), "b": len(b_dups)},
        "label_dist": {
            "a": dict(a_label_cnt),
            "b": dict(b_label_cnt),
        },
        "count_label_conflicts_on_intersection": len(conflicts),
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "only_a.txt").write_text("\n".join(only_a), encoding="utf-8")
    (out_dir / "only_b.txt").write_text("\n".join(only_b), encoding="utf-8")

    if a_dups:
        (out_dir / "dups_a.json").write_text(json.dumps(a_dups, ensure_ascii=False, indent=2), encoding="utf-8")
    if b_dups:
        (out_dir / "dups_b.json").write_text(json.dumps(b_dups, ensure_ascii=False, indent=2), encoding="utf-8")
    if conflicts:
        (out_dir / "label_conflicts.json").write_text(json.dumps(conflicts, ensure_ascii=False, indent=2), encoding="utf-8")

    print("输出目录:", out_dir)
    print("A:", args.a_name, "rows=", len(a_rows), "unique_keys=", len(a_set), "dups=", len(a_dups))
    print("B:", args.b_name, "rows=", len(b_rows), "unique_keys=", len(b_set), "dups=", len(b_dups))
    print("intersection=", len(inter), "only_a=", len(only_a), "only_b=", len(only_b))
    if label_a:
        print("A label_field=", label_a, "dist=", dict(a_label_cnt))
    if label_b:
        print("B label_field=", label_b, "dist=", dict(b_label_cnt))
    if label_a and label_b:
        print("label_conflicts_on_intersection=", len(conflicts))


if __name__ == "__main__":
    main()

