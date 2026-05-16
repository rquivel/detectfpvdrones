"""Auto-assign uncategorized images from source/ into dataset/{train,val}/.

A source image is "uncategorized" if its stem is in source/ but NOT in
dataset/images/train/ or dataset/images/val/ — i.e., it has not been
triaged yet.

The split is *stratified by label class composition*, not random:

  - background       (empty/no label .txt)
  - drone-only       (only class 0 boxes)
  - bird-only        (only class 1 boxes)
  - person-only      (only class 2 boxes)
  - mixed            (multiple classes in the same image)

Each bucket is split independently using the same val ratio, so val ends
up with a representative mix instead of (e.g.) all drones. This matters
because a homogeneous val set gives misleading metrics and triggers
nonsense early-stopping.

Both the .jpg and matching .txt are copied (not moved) into
dataset/images/<split>/ and dataset/labels/<split>/. The source files
are left in place, matching how triage.py works.

Each new assignment is appended to triage.csv so triage.py won't re-iterate
these files later.

Usage:
    python scripts/auto_split.py                 # dry-run, 15% val
    python scripts/auto_split.py --write         # actually copy files
    python scripts/auto_split.py --write --val 0.20
    python scripts/auto_split.py --write --seed 7
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
    """Return a sorted tuple of unique class IDs present in a label file.
    Empty/missing label -> empty tuple = background."""
    if not label_path.exists():
        return ()
    text = label_path.read_text().strip()
    if not text:
        return ()
    ids = set()
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 1:
            try:
                ids.add(int(parts[0]))
            except ValueError:
                pass
    return tuple(sorted(ids))


def bucket_name(class_tuple: tuple[int, ...]) -> str:
    if not class_tuple:
        return "background"
    if len(class_tuple) == 1:
        return CLASS_NAMES.get(class_tuple[0], f"cls{class_tuple[0]}-only")
    names = "+".join(CLASS_NAMES.get(c, f"cls{c}") for c in class_tuple)
    return f"mixed({names})"


def safe_copy(src: Path, dst_dir: Path) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    if not dst.exists():
        shutil.copy2(src, dst)
    return dst


def read_triage_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def append_triage_csv(path: Path, new_rows: list[dict]) -> None:
    existed = path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["image", "decision", "dest", "label_dest"])
        if not existed:
            w.writeheader()
        w.writerows(new_rows)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    here = Path(__file__).resolve().parents[1]
    ap.add_argument("--source", default=str(here / "source"))
    ap.add_argument("--dataset", default=str(here / "dataset"))
    ap.add_argument("--triage-log", default=str(here / "triage.csv"))
    ap.add_argument("--val", type=float, default=0.15,
                    help="Fraction of new images that go to val (default 0.15)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--write", action="store_true",
                    help="Actually copy files and update triage.csv (default: dry-run)")
    args = ap.parse_args()

    source_dir = Path(args.source).resolve()
    dataset_dir = Path(args.dataset).resolve()
    triage_log = Path(args.triage_log).resolve()
    train_img = dataset_dir / "images" / "train"
    train_lbl = dataset_dir / "labels" / "train"
    val_img = dataset_dir / "images" / "val"
    val_lbl = dataset_dir / "labels" / "val"
    for p in (source_dir, dataset_dir):
        if not p.is_dir():
            print(f"error: not a directory: {p}", file=sys.stderr)
            sys.exit(1)

    # Stems already living in dataset/
    already = set()
    for d in (train_img, val_img):
        if d.is_dir():
            for p in d.iterdir():
                if p.is_file() and p.suffix.lower() in IMG_EXTS:
                    already.add(p.stem)

    # Uncategorized source images
    uncategorized: list[Path] = []
    for p in sorted(source_dir.iterdir()):
        if not p.is_file() or p.suffix.lower() not in IMG_EXTS:
            continue
        if p.stem in already:
            continue
        uncategorized.append(p)

    if not uncategorized:
        print("nothing to split: every source image is already in train or val.")
        return

    # Bucket by class composition
    buckets: dict[str, list[Path]] = {}
    for p in uncategorized:
        lbl = p.with_suffix(".txt")
        buckets.setdefault(bucket_name(classes_in_label(lbl)), []).append(p)

    # Stratified split
    rng = random.Random(args.seed)
    plan: list[tuple[Path, str]] = []   # (path, split)
    print(f"source       : {source_dir}")
    print(f"dataset      : {dataset_dir}")
    print(f"uncategorized: {len(uncategorized)} image(s)")
    print(f"val fraction : {args.val}   seed: {args.seed}")
    print()
    print(f"{'bucket':<28}{'total':>8}{'train':>8}{'val':>8}")
    print("-" * 52)
    for bname, items in sorted(buckets.items()):
        rng.shuffle(items)
        n = len(items)
        n_val = max(0, round(n * args.val)) if n > 1 else 0
        # always keep at least one train item per non-empty bucket
        n_val = min(n_val, n - 1)
        val_items = items[:n_val]
        train_items = items[n_val:]
        print(f"{bname:<28}{n:>8}{len(train_items):>8}{len(val_items):>8}")
        for p in train_items:
            plan.append((p, "train"))
        for p in val_items:
            plan.append((p, "val"))
    print("-" * 52)
    total_val = sum(1 for _, s in plan if s == "val")
    print(f"{'TOTAL':<28}{len(plan):>8}{len(plan) - total_val:>8}{total_val:>8}")

    if not args.write:
        print("\ndry-run; re-run with --write to actually copy files and update triage.csv")
        return

    # Apply: copy image + label, append CSV
    new_rows = []
    n_skipped_no_label = 0
    for img_src, split in plan:
        img_dir = train_img if split == "train" else val_img
        lbl_dir = train_lbl if split == "train" else val_lbl
        img_dst = safe_copy(img_src, img_dir)
        lbl_src = img_src.with_suffix(".txt")
        if lbl_src.exists():
            lbl_dst = safe_copy(lbl_src, lbl_dir)
            lbl_dest_str = str(lbl_dst)
        else:
            # No label in source — create an empty one so YOLO treats this
            # as a background image rather than skipping it.
            lbl_dst = lbl_dir / f"{img_src.stem}.txt"
            lbl_dst.parent.mkdir(parents=True, exist_ok=True)
            lbl_dst.touch()
            lbl_dest_str = str(lbl_dst)
            n_skipped_no_label += 1
        new_rows.append({
            "image": img_src.name,
            "decision": split,
            "dest": str(img_dst),
            "label_dest": lbl_dest_str,
        })

    append_triage_csv(triage_log, new_rows)
    print(f"\ncopied {len(new_rows)} pair(s); "
          f"created {n_skipped_no_label} empty label file(s) for missing labels")
    print(f"appended {len(new_rows)} row(s) to {triage_log}")


if __name__ == "__main__":
    main()
