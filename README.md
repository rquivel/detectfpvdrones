# FPV Drone Detection with YOLOv8

A custom-trained YOLOv8 model for detecting FPV (First Person View) drones in images and video. Training is optimized for Apple Silicon (M3 Pro) using the Metal Performance Shaders (MPS) backend.

## Project Overview

- **Architecture:** YOLOv8 Small (`yolov8s.pt`) — upgraded from Nano after the v1 run undertrained the classification head
- **Hardware:** MacBook Pro M3 Pro (MPS GPU acceleration)
- **Classes:** 3 — `drone`, `bird`, `person`. Bird and person are negative-context classes that help the model stop calling everything that flies a drone (off-the-shelf yolov8s classifies FPV drones as `airplane`).
- **Image size:** 640 × 640
- **Dataset split:** 144 training images / 38 validation images

## Repository

```
git@github.com:rquivel/detectfpvdrones.git
```

## Dataset Preparation

The dataset was assembled and labeled locally on macOS.

New candidate images live in `source/` (a staging area, not the training set). The pipeline is:

1. drop new images into `source/`
2. normalize their formats with `convert.sh`
3. label them with YoloLabel
4. run `triage.py` to compare against the latest trained model and split each image into train, val, or skip

### 1. Normalize image formats

YOLO labeling tools work best with `.png` and `.jpg`. [`convert.sh`](convert.sh) batch-converts any `.webp`, `.jpeg`, or `.avif` files in the current directory to PNG using macOS's built-in `sips`:

```bash
cd source
../convert.sh
```

The script removes the originals after a successful conversion.

### 2. Annotate with YoloLabel

A [`classes.txt`](classes.txt) file with three lines (in this order) lives in the project root:

```
drone
bird
person
```

Drop a copy alongside the images you're labeling, then draw bounding boxes in [YoloLabel](https://github.com/developer0hye/Yolo_Label). Each image gets a matching `.txt` file in YOLO format:

```
<class_id> <x_center> <y_center> <width> <height>
```

(All values normalized 0–1 relative to image size.) `class_id` is `0` for drone, `1` for bird, `2` for person — the order **must** match `data.yaml`, otherwise YOLO will reject labels at training time as "corrupt".

### 3. Triage into train / val with `triage.py`

The dataset follows the standard YOLO layout:

```
dataset/
  images/
    train/   # 144 images
    val/     # 38 images
  labels/
    train/   # 144 .txt files
    val/     # 38 .txt files
```

[`triage.py`](triage.py) is an interactive review tool. It opens each image side-by-side with predictions from pretrained `yolov8s.pt` (left) and the most recent locally-trained `best.pt` (right) so you can see how your model handles each candidate before assigning it to a split.

```bash
source venv/bin/activate
python triage.py
```

Keys (one keypress per image):

| Key | Source image (in `source/`) | Image already in dataset |
| --- | --- | --- |
| `t` | copy image + `.txt` label to `dataset/images/train/` and `dataset/labels/train/` | move image + label between splits if needed |
| `v` | copy to `dataset/images/val/` and `dataset/labels/val/` | move between splits if needed |
| `s` | record "skip" in `triage.csv` (no file ops) | ignored — use SPACE |
| `space` | next image, no recording, no file ops | next image, no recording, no file ops |
| `b` | back to previous image | back to previous image |
| `q` | quit | quit |

The originals in `source/` are **never** moved or deleted — only copied. Decisions are appended to [`triage.csv`](triage.csv) so you can quit and resume. After the source queue empties, the tool keeps going through `dataset/images/{train,val}/` so you can review or reorganize already-classified images against the latest trained model.

`--trained` defaults to whichever `runs/detect/*/weights/best.pt` is most recent; pass `--trained PATH` to compare against a specific run. `--restart` wipes `triage.csv` and starts over. `--commit-only` reconciles missing labels for prior decisions (handy if you triage before labeling) and exits without opening the UI.

### 4. `data.yaml`

```yaml
path: /private/var/www/detectfpvdrones/dataset
train: images/train
val: images/val
nc: 3
names: ['drone', 'bird', 'person']
```

If you clone this repo to a different location, update `path` to point at the project's `dataset/` directory.

## Setup

```bash
git clone git@github.com:rquivel/detectfpvdrones.git
cd detectfpvdrones

# Create and activate a virtual environment (required on modern macOS — pip won't install system-wide)
python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

If you don't yet have a `requirements.txt`, install Ultralytics directly and freeze the environment:

```bash
pip install ultralytics
pip freeze > requirements.txt
```

System dependencies (only needed once):

```bash
brew install python ffmpeg
```

## Training

Use the included `train.sh` (activates the venv, runs `yolo` with the tuned hyperparameters):

```bash
./train.sh                # writes to runs/detect/train_v2 (default name)
./train.sh train_v3       # writes to runs/detect/train_v3
```

Equivalent direct command:

```bash
yolo task=detect mode=train \
  model=yolov8s.pt \
  data=data.yaml \
  epochs=200 patience=30 \
  imgsz=640 device=mps \
  cos_lr=True lr0=0.005 \
  name=train_v3
```

> Tip: if you change `data.yaml` (e.g. add a class), delete `dataset/labels/{train,val}.cache` before re-running, or YOLO will reuse the stale "corrupt label" markers from the old class count.

Results, plots, and weights are written to `runs/detect/<name>/`. The best checkpoint is saved at `runs/detect/<name>/weights/best.pt`.

### Why these settings

The first run (`runs/detect/train/`, yolov8n, 50 epochs, default LR) hit a high mAP50 around epoch 35 but the classification head never produced confident predictions — top confidence on the validation set was ~0.025, so `predict(conf=0.25)` returned nothing. The retrain config addresses that:

- **`yolov8s.pt`** — more capacity than nano; helps the cls head separate drone from background on a small dataset.
- **`epochs=200 patience=30`** — long enough to converge, with early stopping so we keep the best checkpoint instead of overfitting.
- **`cos_lr=True lr0=0.005`** — gentler, decaying schedule so the cls head stabilizes near the end of training.

## Results (v3, `runs/detect/train_v3/`)

Three-class run on 144 train / 38 val images (200 epochs, no early stop). Best epoch (192) by `mAP50-95`:

| Precision | Recall | mAP50 | mAP50-95 |
| --------- | ------ | ----- | -------- |
| 0.674     | 0.876  | 0.716 | 0.549    |

GPU memory ~7.7 GB on the M3 Pro at `imgsz=640`.

**Caveat on the headline numbers.** All 26 instances in the validation set are drones — there are zero `bird` and zero `person` instances in `val/`. Bird and person performance is therefore not reflected in `mAP`; the aggregated metrics above are effectively drone-only. Add labeled bird and person samples to `dataset/images/val/` (and run `triage.py` to move them) before trusting per-class metrics.

### Earlier runs (historical)

- **v1** (`runs/detect/train/`, yolov8n, 50 epochs, single-class): mAP50 peaked at 0.97 around epoch 35 then regressed to 0.77 by epoch 50. The classification head never produced confident predictions — top val confidence was ~0.025, so `predict(conf=0.25)` returned nothing.
- **v2** (`runs/detect/train_v2/`, yolov8s, single-class, early-stopped at epoch 32): P=0.85, R=0.875, mAP50=0.812, mAP50-95=0.493. Top val confidences moved into the 0.5–0.84 range, fixing the v1 regression. v3 supersedes this once bird/person samples are in the val set.

## Image inference

Quick Python check (see [`test.py`](test.py)):

```python
from ultralytics import YOLO
model = YOLO('./runs/detect/train_v3/weights/best.pt')
results = model.predict(source='./dataset/images/train/image2.png', conf=0.25)
results[0].show()
```

CLI equivalent:

```bash
yolo task=detect mode=predict \
  model=runs/detect/train_v3/weights/best.pt \
  source='path/to/image_or_video' \
  show=True
```

## Real-time video inference

Use [`video.py`](video.py) for live detection on a webcam, video file, or RTSP/HTTP stream. It opens an OpenCV window with annotated frames, an FPS counter, and a per-frame detection count. Press **q** to quit.

```bash
python3 video.py                                # webcam (source 0)
python3 video.py path/to/fpv_footage.mp4        # local video file
python3 video.py rtsp://192.168.1.10:554/stream # network stream
```

Useful flags:

| Flag | Default | What it does |
| ---- | ------- | ------------ |
| `--weights PATH` | `runs/detect/train_v3/weights/best.pt` | Use a different checkpoint. |
| `--conf FLOAT` | `0.25` | Confidence threshold. Try `0.10`–`0.15` on unfamiliar footage. |
| `--imgsz INT` | `640` | Inference resolution. Bump to `1280` for small/distant drones. |
| `--device` | `mps` | `mps`, `cpu`, or a CUDA index. |
| `--track` | off | Switch from `predict` to `track` so each drone gets a persistent ID across frames. |
| `--save PATH` | off | Also write the annotated video to disk (e.g. `--save annotated.mp4`). |

Example: persistent IDs at a relaxed threshold, saving the result:

```bash
python3 video.py clip.mp4 --conf 0.15 --track --save annotated.mp4
```

## Project Structure

```
detectfpvdrones/
├── data.yaml              # dataset config (nc=3, drone/bird/person)
├── classes.txt            # YoloLabel class order — must match data.yaml
├── train.sh               # retrain entry point
├── convert.sh             # batch sips conversion (webp/jpeg/avif → png)
├── triage.py              # interactive train/val splitter + model comparison
├── triage.csv             # decisions log (auto-generated, safe to delete)
├── test.py                # quick image-inference example
├── video.py               # real-time webcam / video / stream inference
├── requirements.txt       # python dependencies
├── README.md
├── source/                # staging area for new candidate images (untracked)
├── dataset/               # images + labels (gitignored)
│   ├── images/{train,val}
│   └── labels/{train,val}
└── runs/                  # training output (gitignored)
    └── detect/<name>/weights/best.pt
```

## .gitignore

```
venv/
*.pt
runs/
dataset/
source/
triage.csv
__pycache__/
.DS_Store
```

Large binaries (weights, dataset images) and the local triage state are intentionally kept out of the repo.

## Notes

- For faster-moving FPV footage where motion blur hurts detection, try `imgsz=1280`. On the M3 Pro, `imgsz=640` is the sweet spot for speed.
- If `yolo` isn't on your PATH after install, fall back to `python3 -m ultralytics ...`.
- Always `source venv/bin/activate` before running training or inference.
