# FPV Drone Detection with YOLOv8

A custom-trained YOLOv8 model for detecting FPV (First Person View) drones in images and video. Training is optimized for Apple Silicon (M3 Pro) using the Metal Performance Shaders (MPS) backend.

## Project Overview

- **Architecture:** YOLOv8 Nano (`yolov8n.pt`)
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
path: /Users/raphaelquivel/Downloads/droneImages/dataset
train: images/train
val: images/val
nc: 1
names: ["drone"]
```

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

Train with MPS acceleration on Apple Silicon:

```bash
yolo task=detect mode=train \
  model=yolov8n.pt \
  data=data.yaml \
  epochs=50 \
  imgsz=640 \
  device=mps
```

Results, plots, and weights are written to `runs/detect/train/`. The best checkpoint is saved at `runs/detect/train/weights/best.pt`.

## Results

Training reached strong accuracy quickly on this small dataset:

| Epoch | mAP50 |
| ----- | ----- |
| 1     | 0.33  |
| 5     | 0.78  |
| 10    | 0.83  |
| 35    | 0.97  |

GPU memory use stayed around 4.3 GB on the M3 Pro.

## Inference

Run detection on an image or video:

```bash
yolo task=detect mode=predict \
  model=runs/detect/train/weights/best.pt \
  source='path/to/fpv_footage.mp4' \
  show=True
```

## Project Structure

```
detectfpvdrones/
├── data.yaml              # dataset config
├── requirements.txt       # python dependencies
├── README.md
├── dataset/               # images + labels (gitignored)
│   ├── images/{train,val}
│   └── labels/{train,val}
└── runs/                  # training output (gitignored)
    └── detect/train/weights/best.pt
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
