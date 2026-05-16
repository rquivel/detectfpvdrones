"""Rebalance dataset/{train,val} to a target val ratio.

Moves images (and their matching labels) between dataset/images/train/ and
dataset/images/val/ so the val set ends up at the target fraction of the
total. The split is *stratified by label class composition* so val gets a
representative mix of drone/bird/person/background instead of (e.g.) only
drones.

Updates triage.csv in place so triage.py sees the new state.

Run from the repo root:
    python scripts/rebalance_split.py             # dry-run, target 15% val
    python scripts/rebalance_split.py --write     # apply
    python scripts/rebalance_split.py --write --val 0.20
    python scripts/rebalance_split.py --write --seed 7
"""
from __future__ import annotations

import argparse
import csv
import random
import shutil
import sys
from pathlib import Path

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
CLASS_NAMES = {0: "drone", 1: "bird", 2: "person"}


def classes_in_label(label_path: Path) -> tuple[int, ...]:
    if not label_path.exists():
        return ()
    text = label_path.read_text().strip()
    if not text:
        return ()
    ids = set()
    for line in text.splitlines():
        parts = line.split()
        if parts:
            try:
                ids.add(int(parts[0]))
            except ValueError:
                pass
    return tuple(sorted(ids))


def bucket_name(c: tuple[int, ...]) -> str:
    if not c:
        return "background"
    return "+".join(CLASS_NAMES.get(i, f"cls{i}") for i in c)


def collect(images_dir: Path, labels_dir: Path) -> list[tuple[Path, str]]:
    """Return [(image_path, bucket), ...] for one split."""
    out: list[tuple[Path, str]] = []
    if not images_dir.is_dir():
        return out
    for p in sorted(images_dir.iterdir()):
        if not p.is_file() or p.suffix.lower() not in IMG_EXTS:
            continue
        lbl = labels_dir / f"{p.stem}.txt"
        out.append((p, bucket_name(classes_in_label(lbl))))
    return out


def safe_move(src: Path, dst_dir: Path) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    if dst.exists() and dst.resolve() != src.resolve():
        # If dst already has same-name file, treat it as already-in-place
        # and remove the source to avoid duplicates.
        src.unlink()
        return dst
    if src.resolve() != dst.resolve():
        shutil.move(str(src), str(dst))
    return dst


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    here = Path(__file__).resolve().parents[1]
    ap.add_argument("--dataset", default=str(here / "dataset"))
    ap.add_argument("--triage-log", default=str(here / "triage.csv"))
    ap.add_argument("--val", type=float, default=0.15,
                    help="Target val fraction of the total (default 0.15)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--write", action="store_true",
                    help="Apply moves (default: dry-run)")
    args = ap.parse_args()

    dataset = Path(args.dataset).resolve()
    triage_log = Path(args.triage_log).resolve()
    train_img = dataset / "images" / "train"
    train_lbl = dataset / "labels" / "train"
    val_img = dataset / "images" / "val"
    val_lbl = dataset / "labels" / "val"
    if not dataset.is_dir():
        print(f"error: not a directory: {dataset}", file=sys.stderr)
        sys.exit(1)

    train_items = collect(train_img, train_lbl)
    val_items = collect(val_img, val_lbl)
    print(f"dataset      : {dataset}")
    print(f"target val   : {args.val}   seed: {args.seed}")
    print(f"current train: {len(train_items)}")
    print(f"current val  : {len(val_items)}")
    total = len(train_items) + len(val_items)
    overall_target_val = round(total * args.val)
    print(f"target totals: train={total - overall_target_val}  "
          f"val={overall_target_val}\n")

    # Bucket each split
    by_bucket: dict[str, dict[str, list[Path]]] = {}
    for p, b in train_items:
        by_bucket.setdefault(b, {"train": [], "val": []})["train"].append(p)
    for p, b in val_items:
        by_bucket.setdefault(b, {"train": [], "val": []})["val"].append(p)

    rng = random.Random(args.seed)
    plan: list[tuple[Path, str, str]] = []   # (image_path, src_split, dst_split)

    print(f"{'bucket':<28}{'total':>7}{'cur_v':>7}{'tgt_v':>7}{'move':>7}")
    print("-" * 56)
    for bname in sorted(by_bucket):
        b = by_bucket[bname]
        n_train, n_val = len(b["train"]), len(b["val"])
        n = n_train + n_val
        target_val = round(n * args.val)
        # Always keep at least one in each split if both are non-empty,
        # and never move so much that train goes empty.
        if n > 1 and target_val == 0:
            target_val = 0
        target_val = min(target_val, n - 1) if n > 1 else target_val
        delta = target_val - n_val
        print(f"{bname:<28}{n:>7}{n_val:>7}{target_val:>7}{delta:>+7}")

        if delta > 0:
            # Move `delta` images from train -> val
            picks = b["train"][:]
            rng.shuffle(picks)
            for p in picks[:delta]:
                plan.append((p, "train", "val"))
        elif delta < 0:
            # Move -delta images from val -> train
            picks = b["val"][:]
            rng.shuffle(picks)
            for p in picks[:-delta]:
                plan.append((p, "val", "train"))

    print("-" * 56)
    n_t2v = sum(1 for _, s, d in plan if s == "train" and d == "val")
    n_v2t = sum(1 for _, s, d in plan if s == "val" and d == "train")
    print(f"plan: move {n_t2v} train->val   and   {n_v2t} val->train  "
          f"({len(plan)} total)")

    if not args.write:
        print("\ndry-run; re-run with --write to apply")
        return

    if not plan:
        print("\nnothing to move; already at target.")
        return

    # Execute the moves
    moved = []
    for img_src, src_split, dst_split in plan:
        dst_img_dir = val_img if dst_split == "val" else train_img
        dst_lbl_dir = val_lbl if dst_split == "val" else train_lbl
        src_lbl = (train_lbl if src_split == "train" else val_lbl) / f"{img_src.stem}.txt"
        new_img = safe_move(img_src, dst_img_dir)
        new_lbl = safe_move(src_lbl, dst_lbl_dir) if src_lbl.exists() else None
        moved.append((img_src.name, dst_split, str(new_img),
                      str(new_lbl) if new_lbl else ""))

    print(f"\nmoved {len(moved)} pair(s).")

    # Update triage.csv: change the decision/dest for each moved image.
    if triage_log.exists():
        with triage_log.open() as f:
            rows = list(csv.DictReader(f))
    else:
        rows = []
    rows_by_name = {r["image"]: r for r in rows}
    appended = 0
    for name, decision, dest, lbl_dest in moved:
        if name in rows_by_name:
            r = rows_by_name[name]
            r["decision"] = decision
            r["dest"] = dest
            r["label_dest"] = lbl_dest
        else:
            rows.append({"image": name, "decision": decision,
                         "dest": dest, "label_dest": lbl_dest})
            appended += 1
    with triage_log.open("w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["image", "decision", "dest", "label_dest"])
        w.writeheader()
        w.writerows(rows)
    print(f"triage.csv: updated {len(moved) - appended} existing row(s), "
          f"appended {appended} new row(s).")


if __name__ == "__main__":
    main()
