"""Ensure every image in dataset/images/<split>/ has a matching .txt label file.

For images that have no label file, an EMPTY one is created. In YOLO's
training pipeline an empty label file marks the image as a "background"
image — used to teach the model that nothing should be detected there.

Run from the repo root:
    python scripts/ensure_labels.py             # dry-run (default), shows what would happen
    python scripts/ensure_labels.py --write     # actually create the empty .txt files
    python scripts/ensure_labels.py --write --splits train val
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def scan_split(dataset_root: Path, split: str) -> tuple[list[Path], list[Path]]:
    """Return (images, missing_label_paths) for one split."""
    img_dir = dataset_root / "images" / split
    lbl_dir = dataset_root / "labels" / split
    if not img_dir.is_dir():
        return [], []
    lbl_dir.mkdir(parents=True, exist_ok=True)

    images = [p for p in sorted(img_dir.iterdir())
              if p.is_file() and p.suffix.lower() in IMG_EXTS]
    missing = [lbl_dir / f"{p.stem}.txt"
               for p in images
               if not (lbl_dir / f"{p.stem}.txt").exists()]
    return images, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None,
                    help="Path to dataset/ folder (default: ../dataset relative to this script)")
    ap.add_argument("--splits", nargs="+", default=["train", "val"],
                    help="Splits to process (default: train val)")
    ap.add_argument("--write", action="store_true",
                    help="Actually create the empty label files (default: dry-run)")
    args = ap.parse_args()

    if args.dataset:
        dataset_root = Path(args.dataset).resolve()
    else:
        dataset_root = Path(__file__).resolve().parents[1] / "dataset"

    if not dataset_root.is_dir():
        print(f"error: dataset folder not found: {dataset_root}", file=sys.stderr)
        sys.exit(1)

    print(f"dataset: {dataset_root}")
    print(f"mode   : {'WRITE' if args.write else 'dry-run'}")
    total_missing = 0
    for split in args.splits:
        images, missing = scan_split(dataset_root, split)
        print(f"\n== {split} ==")
        print(f"  images : {len(images)}")
        print(f"  missing: {len(missing)} label file(s)")
        if missing:
            print("  first few:")
            for p in missing[:5]:
                print(f"    {p.name}")
            if len(missing) > 5:
                print(f"    ... and {len(missing) - 5} more")
        total_missing += len(missing)

        if args.write:
            for p in missing:
                p.touch()
            if missing:
                print(f"  -> created {len(missing)} empty label file(s)")

    print(f"\nTotal missing across all splits: {total_missing}")
    if not args.write and total_missing:
        print("Re-run with --write to actually create the empty label files.")


if __name__ == "__main__":
    main()
