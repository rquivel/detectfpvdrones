"""Remove dataset/ files whose stem is not present in source/.

source/ is the canonical pool of curated image+label pairs; the YOLO
dataset under dataset/images/{train,val}/ and dataset/labels/{train,val}/
should be a strict subset by stem. This script enforces that by deleting
any dataset image (and its matching label file) whose stem doesn't appear
in source/.

Run from the repo root:
    python scripts/prune_to_source.py             # dry-run, lists what would go
    python scripts/prune_to_source.py --write     # actually delete
    python scripts/prune_to_source.py --splits train val
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def stems_in_source(source_dir: Path) -> set[str]:
    return {
        p.stem
        for p in source_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    }


def find_orphans(dataset_root: Path, split: str,
                 source_stems: set[str]) -> list[tuple[Path, Path | None]]:
    """Return [(image_path, label_path_or_None), ...] for stems not in source."""
    img_dir = dataset_root / "images" / split
    lbl_dir = dataset_root / "labels" / split
    if not img_dir.is_dir():
        return []
    orphans: list[tuple[Path, Path | None]] = []
    for img in sorted(img_dir.iterdir()):
        if not img.is_file() or img.suffix.lower() not in IMG_EXTS:
            continue
        if img.stem in source_stems:
            continue
        label = lbl_dir / f"{img.stem}.txt"
        orphans.append((img, label if label.exists() else None))
    return orphans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None,
                    help="Path to dataset/ folder (default: ../dataset relative to this script)")
    ap.add_argument("--source", default=None,
                    help="Path to source/ folder (default: ../source relative to this script)")
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--write", action="store_true",
                    help="Actually delete files (default: dry-run)")
    args = ap.parse_args()

    here = Path(__file__).resolve().parents[1]
    dataset_root = Path(args.dataset).resolve() if args.dataset else here / "dataset"
    source_dir = Path(args.source).resolve() if args.source else here / "source"

    for p in (dataset_root, source_dir):
        if not p.is_dir():
            print(f"error: not a directory: {p}", file=sys.stderr)
            sys.exit(1)

    source_stems = stems_in_source(source_dir)
    print(f"source : {source_dir}  ({len(source_stems)} image stems)")
    print(f"dataset: {dataset_root}")
    print(f"mode   : {'DELETE' if args.write else 'dry-run'}")

    total_deleted = 0
    for split in args.splits:
        orphans = find_orphans(dataset_root, split, source_stems)
        print(f"\n== {split} ==  ({len(orphans)} orphan image(s))")
        for img, lbl in orphans[:10]:
            print(f"  {img.name}" + (f"   + {lbl.name}" if lbl else "   (no label file)"))
        if len(orphans) > 10:
            print(f"  ... and {len(orphans) - 10} more")

        if args.write:
            for img, lbl in orphans:
                img.unlink(missing_ok=True)
                if lbl is not None:
                    lbl.unlink(missing_ok=True)
            if orphans:
                print(f"  -> deleted {len(orphans)} image(s) + {sum(1 for _, l in orphans if l)} label(s)")
        total_deleted += len(orphans)

    print(f"\nTotal orphans across all splits: {total_deleted}")
    if not args.write and total_deleted:
        print("Re-run with --write to actually delete.")


if __name__ == "__main__":
    main()
