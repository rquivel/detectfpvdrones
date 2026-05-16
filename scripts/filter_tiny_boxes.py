"""Drop labels whose bounding box is too small for YOLO to learn reliably.

At imgsz=640, YOLO's smallest feature-map stride is 8 px, so any box much
smaller than ~16x16 px (area ~0.0006 of image area) is below the model's
detection floor. Training on those is mostly noise — they inflate box_loss
without teaching anything useful.

This script edits .txt label files in place:
    - lines with w*h < --min-area are removed
    - if all lines in a file are removed, the file becomes empty (= background
      image, which is still useful — see ensure_labels.py)

Run from the repo root:
    python scripts/filter_tiny_boxes.py                   # dry-run, threshold 0.001
    python scripts/filter_tiny_boxes.py --write
    python scripts/filter_tiny_boxes.py --write --min-area 0.0005
    python scripts/filter_tiny_boxes.py --write --splits train val
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def process(label_path: Path, min_area: float):
    """Return (kept_lines, dropped_count) for a label file."""
    text = label_path.read_text()
    if not text.strip():
        return [], 0
    kept = []
    dropped = 0
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 5:
            kept.append(line)   # preserve unexpected formats untouched
            continue
        try:
            w = float(parts[3]); h = float(parts[4])
        except ValueError:
            kept.append(line)
            continue
        if w * h >= min_area:
            kept.append(line)
        else:
            dropped += 1
    return kept, dropped


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    here = Path(__file__).resolve().parents[1]
    ap.add_argument("--dataset", default=str(here / "dataset"))
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--min-area", type=float, default=0.001,
                    help="Minimum w*h to keep. 0.001 = 0.1%% of image, "
                         "~20px on a side at imgsz=640. (Default: 0.001)")
    ap.add_argument("--write", action="store_true",
                    help="Actually edit files (default: dry-run)")
    args = ap.parse_args()

    dataset = Path(args.dataset).resolve()
    print(f"dataset : {dataset}")
    print(f"min area: {args.min_area}  ({args.min_area*100:.3f}% of image)")
    print(f"mode    : {'WRITE' if args.write else 'dry-run'}\n")

    grand_total_files = 0
    grand_files_changed = 0
    grand_files_emptied = 0
    grand_boxes_dropped = 0

    for split in args.splits:
        lbl_dir = dataset / "labels" / split
        if not lbl_dir.is_dir():
            print(f"== {split} ==  (no such folder, skipping)")
            continue

        files = sorted(p for p in lbl_dir.glob("*.txt") if p.name != "classes.txt")
        n_files = len(files)
        n_changed = 0
        n_emptied = 0
        n_dropped = 0
        before_lines = 0
        after_lines = 0

        for lbl in files:
            text = lbl.read_text()
            if not text.strip():
                continue
            kept, dropped = process(lbl, args.min_area)
            orig = [ln for ln in text.splitlines() if ln.strip()]
            before_lines += len(orig)
            after_lines += len(kept)
            if dropped > 0:
                n_changed += 1
                n_dropped += dropped
                if not kept:
                    n_emptied += 1
                if args.write:
                    lbl.write_text(("\n".join(kept) + "\n") if kept else "")

        print(f"== {split} ==")
        print(f"  label files       : {n_files}")
        print(f"  boxes before      : {before_lines}")
        print(f"  boxes dropped     : {n_dropped} "
              f"({100*n_dropped/before_lines:.1f}%)" if before_lines else "")
        print(f"  boxes after       : {after_lines}")
        print(f"  files modified    : {n_changed}")
        print(f"  files now empty   : {n_emptied} (became background images)")
        print()

        grand_total_files += n_files
        grand_files_changed += n_changed
        grand_files_emptied += n_emptied
        grand_boxes_dropped += n_dropped

    print(f"== total ==")
    print(f"  files modified : {grand_files_changed} / {grand_total_files}")
    print(f"  files emptied  : {grand_files_emptied}")
    print(f"  boxes dropped  : {grand_boxes_dropped}")
    if not args.write:
        print("\ndry-run; re-run with --write to actually edit files")


if __name__ == "__main__":
    main()
