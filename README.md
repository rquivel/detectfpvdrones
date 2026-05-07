# FPV Drone Detection with YOLOv8

A custom-trained YOLOv8 model for detecting FPV (First Person View) drones in images and video. Training is optimized for Apple Silicon (M3 Pro) using the Metal Performance Shaders (MPS) backend.

## Project Overview

- **Architecture:** YOLOv8 Small (`yolov8s.pt`) — upgraded from Nano after the v1 run undertrained the classification head
- **Hardware:** MacBook Pro M3 Pro (MPS GPU acceleration)
- **Classes:** 1 (`drone`)
- **Image size:** 640 × 640
- **Dataset split:** 36 training images / 9 validation images

## Repository

```
git@github.com:rquivel/detectfpvdrones.git
```

## Dataset Preparation

The dataset was assembled and labeled locally on macOS.

### 1. Normalize image formats

YOLO labeling tools work best with `.png` and `.jpg`. Any `.webp` or `.jpeg` images were converted using macOS's built-in `sips`:

```bash
cd ~/Downloads/droneImages
for i in *.webp; do sips -s format png "$i" --out "${i%.webp}.png"; done
for i in *.jpeg *.jpg; do sips -s format png "$i" --out "${i%.*}.png"; done
rm *.webp *.jpeg
```

### 2. Annotate with YoloLabel

A `classes.txt` file containing a single line `drone` was placed in the image directory, then bounding boxes were drawn in [YoloLabel](https://github.com/developer0hye/Yolo_Label). Each image gets a matching `.txt` file in YOLO format:

```
<class_id> <x_center> <y_center> <width> <height>
```

(All values normalized 0–1 relative to image size.)

### 3. Split into train / val

The dataset follows the standard YOLO layout:

```
dataset/
  images/
    train/   # 36 images
    val/     # 9 images
  labels/
    train/   # 36 .txt files
    val/     # 9 .txt files
```

A small Python one-liner was used to randomly select 9 images for the validation set (since macOS doesn't ship with `shuf`):

```bash
cd dataset/images/train
python3 -c "import os, random; files=[f for f in os.listdir('.') if f.lower().endswith(('.png','.jpg','.jpeg'))]; random.shuffle(files); [print(f) for f in files[:9]]" \
  | while read -r file; do
      base="${file%.*}"
      mv "$file" ../val/
      mv "../../labels/train/$base.txt" "../../labels/val/" 2>/dev/null
    done
cd ../../../
```

### 4. `data.yaml`

```yaml
path: /private/var/www/detectfpvdrones/dataset
train: images/train
val: images/val
nc: 1
names: ["drone"]
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
./train.sh                # writes to runs/detect/train_v2
./train.sh my_run_name    # writes to runs/detect/my_run_name
```

Equivalent direct command:

```bash
yolo task=detect mode=train \
  model=yolov8s.pt \
  data=data.yaml \
  epochs=200 patience=30 \
  imgsz=640 device=mps \
  cos_lr=True lr0=0.005 \
  name=train_v2
```

Results, plots, and weights are written to `runs/detect/<name>/`. The best checkpoint is saved at `runs/detect/<name>/weights/best.pt`.

### Why these settings

The first run (`runs/detect/train/`, yolov8n, 50 epochs, default LR) hit a high mAP50 around epoch 35 but the classification head never produced confident predictions — top confidence on the validation set was ~0.025, so `predict(conf=0.25)` returned nothing. The retrain config addresses that:

- **`yolov8s.pt`** — more capacity than nano; helps the cls head separate drone from background on a small dataset.
- **`epochs=200 patience=30`** — long enough to converge, with early stopping so we keep the best checkpoint instead of overfitting.
- **`cos_lr=True lr0=0.005`** — gentler, decaying schedule so the cls head stabilizes near the end of training.

## Results (v1, `runs/detect/train/`)

Honest numbers from `results.csv`:

| Epoch | Precision | Recall | mAP50 | mAP50-95 |
| ----- | --------- | ------ | ----- | -------- |
| 1     | 0.003     | 1.000  | 0.330 | 0.141    |
| 5     | 0.003     | 0.875  | 0.775 | 0.482    |
| 10    | 0.003     | 1.000  | 0.828 | 0.436    |
| 35    | 0.860     | 0.875  | 0.971 | 0.317    |
| 50    | 0.976     | 0.625  | 0.773 | 0.368    |

GPU memory use stayed around 4.3 GB on the M3 Pro. Note the mAP50 *dropped* from 0.97 at epoch 35 to 0.77 at epoch 50 — early stopping in v2 prevents that regression.

## Results (v2, `runs/detect/train_v2/`)

Validation metrics on `best.pt`:

| Precision | Recall | mAP50 | mAP50-95 |
| --------- | ------ | ----- | -------- |
| 0.85      | 0.875  | 0.812 | 0.493    |

Training stopped early at epoch 32 (best epoch 2 — partly an artifact of the 9-image val set). The key win over v1 isn't the headline metric, it's the **classification confidence**: top val confidences are now in the 0.5–0.84 range vs. 0.025 in v1, so `predict(conf=0.25)` actually returns boxes.

## Image inference

Quick Python check (see [`test.py`](test.py)):

```python
from ultralytics import YOLO
model = YOLO('./runs/detect/train_v2/weights/best.pt')
results = model.predict(source='./dataset/images/train/image2.png', conf=0.25)
results[0].show()
```

CLI equivalent:

```bash
yolo task=detect mode=predict \
  model=runs/detect/train_v2/weights/best.pt \
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
| `--weights PATH` | `runs/detect/train_v2/weights/best.pt` | Use a different checkpoint. |
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
├── data.yaml              # dataset config
├── train.sh               # retrain entry point
├── test.py                # quick image-inference example
├── video.py               # real-time webcam / video / stream inference
├── requirements.txt       # python dependencies
├── README.md
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
__pycache__/
.DS_Store
```

Large binaries (weights, dataset images) are intentionally kept out of the repo.

## Notes

- For faster-moving FPV footage where motion blur hurts detection, try `imgsz=1280`. On the M3 Pro, `imgsz=640` is the sweet spot for speed.
- If `yolo` isn't on your PATH after install, fall back to `python3 -m ultralytics ...`.
- Always `source venv/bin/activate` before running training or inference.
