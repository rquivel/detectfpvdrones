"""Clean up source/ by removing files whose labels are mostly garbage.

For each <stem>.jpg / <stem>.txt pair in source/, this script:

  1. Reads the .txt label.
  2. Drops lines whose bounding box is smaller than --min-area
     (default 0.001 = 0.1% of image area, ~20x20 px at 640 imgsz).
  3. If ANY lines survive  -> rewrites the .txt with only the kept lines
                              and leaves the image in place.
     If NO lines survive   -> deletes both the .jpg and the .txt.
     If file was empty     -> left alone (it's a background image,
                              still useful).

Run from the repo root (dry-run is the default):

    python scripts/clean_source.py
    python scripts/clean_source.py --write
    python scripts/clean_source.py --write --min-area 0.0005
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def classify(label_path: Path, min_area: float) -> tuple[str, list[str], int]:
    """Return (action, kept_lines, dropped_count).

    action is one of:
        "leave"     — file is empty or all boxes pass; nothing to do
        "trim"      — at least one box was dropped, but some survive
        "delete"    — every box was dropped; file should go away
    """
    text = label_path.read_text() if label_path.exists() else ""
    if not text.strip():
        return "leave", [], 0
    kept: list[str] = []
    dropped = 0
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 5:
            kept.append(line)
            continue
        try:
            w, h = float(parts[3]), float(parts[4])
        except ValueError:
            kept.append(line)
            continue
        if w * h >= min_area:
            kept.append(line)
        else:
            dropped += 1
    if dropped == 0:
        return "leave", kept, 0
    if not kept:
        return "delete", [], dropped
    return "trim", kept, dropped


def find_image_for(stem: str, source_dir: Path) -> Path | None:
    for ext in IMG_EXTS:
        p = source_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    here = Path(__file__).resolve().parents[1]
    ap.add_argument("--source", default=str(here / "source"))
    ap.add_argument("--min-area", type=float, default=0.001,
                    help="Drop boxes smaller than this fraction of the image. "
                         "Default 0.001 = 0.1%% of image area.")
    ap.add_argument("--write", action="store_true",
                    help="Actually delete / rewrite files (default: dry-run)")
    args = ap.parse_args()

    source = Path(args.source).resolve()
    if not source.is_dir():
        print(f"error: source folder not found: {source}", file=sys.stderr)
        sys.exit(1)

    print(f"source  : {source}")
    print(f"min area: {args.min_area}  ({args.min_area*100:.3f}% of image)")
    print(f"mode    : {'WRITE' if args.write else 'dry-run'}\n")

    images = [p for p in sorted(source.iterdir())
              if p.is_file() and p.suffix.lower() in IMG_EXTS]
    if not images:
        print("no images found in source/.")
        return

    n_total = len(images)
    leave = trim = delete = no_label = 0
    deleted_examples: list[str] = []
    trimmed_examples: list[str] = []

    for img in images:
        lbl = img.with_suffix(".txt")
        if not lbl.exists():
            no_label += 1
            continue
        action, kept, dropped = classify(lbl, args.min_area)
        if action == "leave":
            leave += 1
            continue
        if action == "trim":
            trim += 1
            if len(trimmed_examples) < 5:
                trimmed_examples.append(
                    f"{img.name}  (kept {len(kept)}, dropped {dropped})")
            if args.write:
                lbl.write_text("\n".join(kept) + "\n")
            continue
        # action == "delete"
        delete += 1
        if len(deleted_examples) < 5:
            deleted_examples.append(f"{img.name}  (dropped {dropped} tiny box(es))")
        if args.write:
            img.unlink(missing_ok=True)
            lbl.unlink(missing_ok=True)

    print(f"== summary ==")
    print(f"  total images       : {n_total}")
    print(f"  no .txt label      : {no_label}  (left alone)")
    print(f"  kept as-is         : {leave}")
    print(f"  trimmed (label edited): {trim}")
    print(f"  DELETED (img + .txt)  : {delete}")
    if deleted_examples:
        print("\n  example deletions:")
        for e in deleted_examples:
            print(f"    {e}")
        if delete > len(deleted_examples):
            print(f"    ... and {delete - len(deleted_examples)} more")
    if trimmed_examples:
        print("\n  example trims:")
        for e in trimmed_examples:
            print(f"    {e}")

    if not args.write:
        print("\ndry-run; re-run with --write to actually edit/delete files")
    else:
        print(f"\nNext steps:")
        print(f"  1. python scripts/prune_to_source.py --write   "
              f"# cascade source deletions into dataset/")
        print(f"  2. python scripts/rebalance_split.py --write   "
              f"# fix train/val ratio (target 15% val)")
        print(f"  3. python scripts/auto_split.py --write        "
              f"# pick up any newly-uncategorized stems")


if __name__ == "__main__":
    main()
