#!/usr/bin/env python3
"""Side-by-side review of candidate and already-classified images.

Compares pretrained yolov8s.pt against the latest trained best.pt and lets
you decide whether each image goes into train, val, or is skipped.

Iteration order:
  1. images in the source folder that haven't been split yet
  2. images already in dataset/images/train
  3. images already in dataset/images/val

For images coming from source: the original is left in place; the image
and its matching .txt label are COPIED into dataset/images/<split>/ and
dataset/labels/<split>/.

For images already in dataset: t/v MOVE the image and label between
splits (a file can't be in both train and val).

Keys:
  t / v   classify as train / val
  s       skip (record decision in triage.csv; no file ops)
  space   next without recording any change
  b       back to the previous image
  q       quit
"""
import argparse
import csv
import shutil
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

PROJECT = Path(__file__).resolve().parent
DEFAULT_SOURCE = PROJECT / "source"
DEFAULT_DATASET = PROJECT / "dataset"
DEFAULT_LOG = PROJECT / "triage.csv"
DEFAULT_PRETRAINED = PROJECT / "yolov8s.pt"

PANEL_W = 720
WINDOW = "triage  —  t=train  v=val  s=skip  space=next  b=back  q=quit"
IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


@dataclass
class Paths:
    source: Path
    train_img: Path
    train_lbl: Path
    val_img: Path
    val_lbl: Path


def find_latest_trained() -> Path | None:
    """Return the most recent best.pt in runs/detect/*/weights/."""
    runs = PROJECT / "runs" / "detect"
    if not runs.exists():
        return None
    candidates = []
    for d in runs.iterdir():
        if not d.is_dir():
            continue
        best = d / "weights" / "best.pt"
        if best.exists():
            candidates.append((best.stat().st_mtime, best))
    if not candidates:
        return None
    candidates.sort()
    return candidates[-1][1]


def safe_copy(src: Path, dst_dir: Path) -> Path:
    """Copy src into dst_dir. If a file with the same name already exists,
    leave it in place and return its path."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    if not dst.exists():
        shutil.copy2(src, dst)
    return dst


def safe_move(src: Path, dst_dir: Path) -> Path:
    """Move src into dst_dir. If dst already has the file, drop the source."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    if dst.exists() and dst.resolve() != src.resolve():
        src.unlink()
        return dst
    if src.resolve() != dst.resolve():
        shutil.move(str(src), str(dst))
    return dst


def dest_dirs(decision: str, paths: Paths) -> tuple[Path, Path]:
    if decision == "train":
        return paths.train_img, paths.train_lbl
    if decision == "val":
        return paths.val_img, paths.val_lbl
    raise ValueError(f"no destination for decision: {decision}")


def label_for(image_path: Path, location: str, paths: Paths) -> Path:
    """Return the .txt label path matching image_path at its current location."""
    if location == "source":
        return image_path.with_suffix(".txt")
    if location == "train":
        return paths.train_lbl / (image_path.stem + ".txt")
    if location == "val":
        return paths.val_lbl / (image_path.stem + ".txt")
    raise ValueError(f"unknown location: {location}")


def commit(image_path: Path, location: str, decision: str,
           paths: Paths) -> tuple[Path, str, str]:
    """Apply a decision. Returns (new_image_path, image_dest_str, label_dest_str).
    new_image_path is where the file lives after this call (used to update
    history/queue references).
    """
    if decision == "skip":
        return image_path, "", ""

    label_path = label_for(image_path, location, paths)
    img_dir, lbl_dir = dest_dirs(decision, paths)

    if location == "source":
        # copy from source, leave original alone
        new_img = safe_copy(image_path, img_dir)
        new_lbl = safe_copy(label_path, lbl_dir) if label_path.exists() else None
        # the file we still iterate from is the source one
        next_path = image_path
    elif location == decision:
        # already in the right split — no-op
        new_img = image_path
        new_lbl = label_path if label_path.exists() else None
        next_path = image_path
    else:
        # move between dataset splits (and matching label)
        new_img = safe_move(image_path, img_dir)
        new_lbl = safe_move(label_path, lbl_dir) if label_path.exists() else None
        next_path = new_img

    if location == "source" and not new_lbl:
        print(f"note: no label found for {image_path.name}")

    return next_path, str(new_img), str(new_lbl) if new_lbl else ""


def reconcile(rows: list[dict], paths: Paths) -> int:
    """For prior train/val rows whose label hasn't been copied yet, copy it
    from the source folder. Never moves or deletes anything."""
    fixed = 0
    for r in rows:
        decision = r["decision"]
        if decision not in ("train", "val"):
            continue
        if r.get("label_dest") and Path(r["label_dest"]).exists():
            continue
        src_lbl = (paths.source / r["image"]).with_suffix(".txt")
        if not src_lbl.exists():
            continue
        _, lbl_dir = dest_dirs(decision, paths)
        r["label_dest"] = str(safe_copy(src_lbl, lbl_dir))
        fixed += 1
    return fixed


def gather_images(paths: Paths,
                  decisions: dict) -> list[tuple[Path, str]]:
    """Build the review queue: untriaged source images, then dataset
    images (train then val). De-duped by filename."""
    train = ({p.name: p for p in paths.train_img.iterdir()
              if p.is_file() and p.suffix.lower() in IMG_EXTS}
             if paths.train_img.exists() else {})
    val = ({p.name: p for p in paths.val_img.iterdir()
            if p.is_file() and p.suffix.lower() in IMG_EXTS}
           if paths.val_img.exists() else {})

    leak = sorted(set(train) & set(val))
    if leak:
        sample = ", ".join(leak[:3])
        more = f" (+{len(leak) - 3} more)" if len(leak) > 3 else ""
        print(f"warning: {len(leak)} image(s) in BOTH train and val: "
              f"{sample}{more}")

    out: list[tuple[Path, str]] = []
    if paths.source.exists():
        for p in sorted(paths.source.iterdir()):
            if not p.is_file() or p.suffix.lower() not in IMG_EXTS:
                continue
            if p.name in train or p.name in val:
                continue
            if p.name in decisions:
                continue
            out.append((p, "source"))
    for name in sorted(train):
        out.append((train[name], "train"))
    for name in sorted(val):
        if name in train:
            continue
        out.append((val[name], "val"))
    return out


def read_log(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r.setdefault("dest", "")
        r.setdefault("label_dest", "")
    return rows


def write_log(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["image", "decision", "dest", "label_dest"])
        w.writeheader()
        w.writerows(rows)


def fit(img: np.ndarray, w: int) -> np.ndarray:
    h0, w0 = img.shape[:2]
    return cv2.resize(img, (w, int(h0 * (w / w0))))


def banner(img: np.ndarray, title: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(out, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2, cv2.LINE_AA)
    return out


def status(width: int, lines: list[str]) -> np.ndarray:
    h = 22 * len(lines) + 14
    bar = np.zeros((h, width, 3), dtype=np.uint8)
    for i, line in enumerate(lines):
        cv2.putText(bar, line, (8, 22 + i * 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return bar


def pad_to(img: np.ndarray, h: int) -> np.ndarray:
    if img.shape[0] >= h:
        return img
    pad = np.zeros((h - img.shape[0], img.shape[1], 3), dtype=np.uint8)
    return np.vstack([img, pad])


def render(pre_img, post_img, trained_label, header_lines):
    left = fit(pre_img, PANEL_W)
    right = fit(post_img, PANEL_W)
    h = max(left.shape[0], right.shape[0])
    panels = np.hstack([
        banner(pad_to(left, h), "yolov8s.pt  (pretrained, COCO classes)"),
        banner(pad_to(right, h), trained_label),
    ])
    return np.vstack([status(panels.shape[1], header_lines), panels])


def summarize(result, top: int = 3) -> str:
    if result.boxes is None or len(result.boxes) == 0:
        return "(no detections)"
    confs = result.boxes.conf.cpu().numpy()
    cls = result.boxes.cls.cpu().numpy().astype(int)
    order = np.argsort(-confs)[:top]
    return ", ".join(
        f"{result.names[cls[i]]} {confs[i]:.2f}" for i in order)


def report(rows: list[dict]) -> None:
    counts = {"train": 0, "val": 0, "skip": 0}
    for r in rows:
        counts[r["decision"]] = counts.get(r["decision"], 0) + 1
    print(f"\ntriage.csv:  train={counts['train']}  "
          f"val={counts['val']}  skip={counts['skip']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET,
                    help="root with images/{train,val} and labels/{train,val}")
    ap.add_argument("--pretrained", type=Path, default=DEFAULT_PRETRAINED)
    ap.add_argument("--trained", type=Path, default=None,
                    help="trained best.pt; defaults to the most recent run")
    ap.add_argument("--log", type=Path, default=DEFAULT_LOG)
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--restart", action="store_true",
                    help="ignore existing triage.csv and start over")
    ap.add_argument("--commit-only", action="store_true",
                    help="copy any missing labels for prior decisions and exit")
    args = ap.parse_args()

    paths = Paths(
        source=args.source,
        train_img=args.dataset / "images" / "train",
        train_lbl=args.dataset / "labels" / "train",
        val_img=args.dataset / "images" / "val",
        val_lbl=args.dataset / "labels" / "val",
    )

    trained = args.trained or find_latest_trained()
    if not trained:
        sys.exit("no trained model found in runs/detect/*/weights/best.pt; "
                 "pass --trained")
    if not args.pretrained.exists():
        sys.exit(f"pretrained weights not found: {args.pretrained}")
    if not paths.source.exists():
        sys.exit(f"source folder not found: {paths.source}")
    if not trained.exists():
        sys.exit(f"trained weights not found: {trained}")

    print(f"trained model: {trained.relative_to(PROJECT)}")

    if args.restart and args.log.exists():
        args.log.unlink()
    rows = read_log(args.log)

    fixed = reconcile(rows, paths)
    if fixed:
        print(f"reconciled {fixed} prior decision(s): copied missing labels.")
    write_log(args.log, rows)

    if args.commit_only:
        report(rows)
        return

    decisions = {r["image"]: r for r in rows}
    items = gather_images(paths, decisions)
    if not items:
        print("nothing to review.")
        report(rows)
        return

    n_source = sum(1 for _, where in items if where == "source")
    print(f"loading models...   {len(items)} image(s) "
          f"({n_source} new from source, {len(items) - n_source} from dataset).")
    m_pre = YOLO(str(args.pretrained))
    m_post = YOLO(str(trained))
    trained_label = f"{trained.parent.parent.name}/best.pt  (trained)"

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    queue: deque[tuple[Path, str]] = deque(items)
    history: list[tuple[Path, str]] = []

    while queue:
        current, location = queue[0]
        if not current.exists():
            queue.popleft()
            continue
        r_pre = m_pre.predict(source=str(current), conf=args.conf,
                              verbose=False)[0]
        r_post = m_post.predict(source=str(current), conf=args.conf,
                                verbose=False)[0]

        prior = decisions.get(current.name)
        prior_str = (f"  decision={prior['decision']}"
                     if prior else "  decision=-")
        idx = len(history) + 1
        total = len(history) + len(queue)
        header = [
            f"[{idx}/{total}]  {current.name}  [in: {location}]"
            f"{prior_str}    conf>={args.conf:.2f}",
            f"yolov8s : {summarize(r_pre)}",
            f"trained : {summarize(r_post)}",
            "t=Train   v=Val   s=Skip   SPACE=Next   b=Back   q=Quit",
        ]
        cv2.imshow(WINDOW,
                   render(r_pre.plot(), r_post.plot(), trained_label, header))

        key = cv2.waitKey(0) & 0xFF
        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break

        if key == ord("q"):
            break
        if key == ord("b"):
            if not history:
                continue
            queue.appendleft(history.pop())
            continue
        if key == ord(" "):
            history.append((current, location))
            queue.popleft()
            continue

        ch = chr(key) if key < 128 else ""
        decision = {"t": "train", "v": "val", "s": "skip"}.get(ch)
        if decision is None:
            continue

        if decision == "skip" and location != "source":
            print(f"skip ignored for {current.name} (already in {location}); "
                  f"use SPACE to navigate or t/v to move it.")
            continue

        try:
            new_path, img_dest, lbl_dest = commit(
                current, location, decision, paths)
        except Exception as e:
            print(f"failed to {decision} {current.name}: {e}")
            continue

        if (prior and prior["decision"] in ("train", "val")
                and decision != prior["decision"]
                and location == "source"):
            print(f"note: previous {prior['decision']} copy of "
                  f"{current.name} at {prior['dest']} was NOT removed.")

        new_row = {"image": current.name, "decision": decision,
                   "dest": img_dest, "label_dest": lbl_dest}
        if prior:
            rows = [new_row if r["image"] == current.name else r
                    for r in rows]
        else:
            rows.append(new_row)
        decisions[current.name] = new_row
        write_log(args.log, rows)

        # The new "location" for history is wherever the file ended up.
        new_loc = decision if decision in ("train", "val") else location
        history.append((new_path, new_loc))
        queue.popleft()

    cv2.destroyAllWindows()
    report(rows)


if __name__ == "__main__":
    main()
