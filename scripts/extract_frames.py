"""Extract frames from a video file so they can be labeled.

Output goes to a *staging* folder (default: extracted/<video_stem>/) so
nothing pollutes source/ or dataset/ until you've picked which frames are
worth labeling. Filenames embed the timestamp, e.g.
'testVideo2_t000.50s.jpg', so you can always trace a frame back to its
source video.

Usage:
    # Extract 1 frame per second (default)
    python scripts/extract_frames.py testVideo2.mp4

    # Extract 2 frames per second, capped at 200 total
    python scripts/extract_frames.py testVideo2.mp4 --fps 2 --max-frames 200

    # Only the first minute of the video
    python scripts/extract_frames.py testVideo2.mp4 --start 0 --end 60

    # Custom output folder
    python scripts/extract_frames.py testVideo2.mp4 --output-dir /tmp/frames

After extraction, the typical YOLO labeling workflow:
    1. Open the output folder in YoloLabel.app
    2. Draw boxes / delete unusable frames
    3. Move the kept .jpg + .txt pairs into source/
    4. Run scripts/ensure_labels.py to backfill empty labels if needed
    5. Re-run triage.py to assign each into train or val
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("video", help="Path to a video file (mp4, mov, avi, ...)")
    ap.add_argument("--output-dir", default=None,
                    help="Where to write JPGs (default: extracted/<video_stem>/)")
    ap.add_argument("--fps", type=float, default=1.0,
                    help="Frames per second to extract (default: 1.0 = one frame "
                         "per second of video)")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="Cap the number of frames written (default: unlimited)")
    ap.add_argument("--start", type=float, default=0.0,
                    help="Skip to this many seconds before extracting (default: 0)")
    ap.add_argument("--end", type=float, default=None,
                    help="Stop extracting after this many seconds (default: end of video)")
    ap.add_argument("--quality", type=int, default=95,
                    help="JPEG quality 1-100 (default: 95)")
    ap.add_argument("--prefix", default=None,
                    help="Filename prefix (default: derived from video filename)")
    args = ap.parse_args()

    src = Path(args.video)
    if not src.exists():
        print(f"error: {src} does not exist", file=sys.stderr)
        sys.exit(1)

    project_root = Path(__file__).resolve().parents[1]
    out_dir = (Path(args.output_dir)
               if args.output_dir
               else project_root / "extracted" / src.stem)
    out_dir.mkdir(parents=True, exist_ok=True)

    prefix = args.prefix or src.stem.replace(" ", "_")

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        print(f"error: could not open video: {src}", file=sys.stderr)
        sys.exit(1)

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / video_fps if video_fps else 0.0

    # How many source frames to skip between captures.
    # e.g. video_fps=30, args.fps=1 -> step=30 (one source frame every 30)
    step = max(1, int(round(video_fps / args.fps)))

    start_frame = int(args.start * video_fps)
    end_frame = (int(args.end * video_fps)
                 if args.end is not None
                 else total_frames)

    print(f"Source : {src}")
    print(f"  {width}x{height} @ {video_fps:.2f} fps, "
          f"{total_frames} frames ({duration:.1f}s)")
    print(f"Output : {out_dir}")
    print(f"  every {step} source frame(s) "
          f"-> {args.fps:g} extracted fps")
    if args.max_frames:
        print(f"  capped at {args.max_frames} output frames")

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    written = 0
    frame_idx = start_frame
    jpeg_params = [cv2.IMWRITE_JPEG_QUALITY, max(1, min(args.quality, 100))]

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx >= end_frame:
                break
            # Only write every Nth frame starting from start_frame
            if (frame_idx - start_frame) % step == 0:
                t = frame_idx / video_fps if video_fps else 0.0
                name = f"{prefix}_t{t:08.2f}s.jpg"
                if cv2.imwrite(str(out_dir / name), frame, jpeg_params):
                    written += 1
                    if args.max_frames and written >= args.max_frames:
                        break
                else:
                    print(f"warning: failed to write {name}", file=sys.stderr)
            frame_idx += 1
    finally:
        cap.release()

    print(f"\nWrote {written} frame(s) to {out_dir}")
    print("\nNext steps:")
    print(f"  1. Review frames in {out_dir} (Finder -> spacebar previews)")
    print( "  2. Delete duplicates / unusable frames")
    print( "  3. Open the folder in YoloLabel.app and draw boxes")
    print(f"  4. Move kept .jpg + matching .txt into source/ :")
    print(f"       mv {out_dir}/*.{{jpg,txt}} source/")
    print( "  5. Run triage.py to assign each into train or val")


if __name__ == "__main__":
    main()
