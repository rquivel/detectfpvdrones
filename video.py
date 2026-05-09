import argparse
import time

import cv2
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Real-time drone detection on a video or webcam.")
    parser.add_argument("source", nargs="?", default="0",
                        help="Path to a video file, an RTSP/HTTP URL, or a webcam index (default: 0).")
    parser.add_argument("--weights", default="./runs/detect/train_v3/weights/best.pt",
                        help="Path to YOLO weights.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--device", default="mps", help="Inference device (mps, cpu, 0, ...).")
    parser.add_argument("--save", metavar="PATH",
                        help="Optional path to write the annotated video (e.g. out.mp4).")
    parser.add_argument("--track", action="store_true",
                        help="Use ByteTrack/BoTSORT to assign persistent IDs across frames.")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source

    model = YOLO(args.weights)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {source}")
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, fps_in, (width, height))

    runner = model.track if args.track else model.predict
    kwargs = dict(source=source, conf=args.conf, imgsz=args.imgsz,
                  device=args.device, stream=True, verbose=False)
    if args.track:
        kwargs["persist"] = True

    window = "drones (q to quit)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    last = time.time()
    smoothed_fps = 0.0
    try:
        for result in runner(**kwargs):
            frame = result.plot()

            now = time.time()
            inst_fps = 1.0 / max(now - last, 1e-6)
            smoothed_fps = 0.9 * smoothed_fps + 0.1 * inst_fps if smoothed_fps else inst_fps
            last = now
            cv2.putText(frame, f"{smoothed_fps:5.1f} FPS  {len(result.boxes)} det",
                        (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

            if writer is not None:
                writer.write(frame)
            cv2.imshow(window, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
