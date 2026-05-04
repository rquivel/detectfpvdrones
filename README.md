FPV Drone Detection with YOLOv8This repository contains a custom-trained YOLOv8 model designed to detect FPV (First Person View) drones. The project is optimized for Apple Silicon (M3 Pro) using the Metal Performance Shaders (MPS) backend for hardware-accelerated training and inference.  🚀 Quick Start1. Environment SetupClone the repository and set up the Python virtual environment:  Bashgit clone git@github.com:rquivel/detectfpvdrones.git
cd detectfpvdrones
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
2. Training on macOS (MPS)The model is trained using the yolov8n.pt (nano) weights to ensure high FPS during real-time detection. To start training with your local dataset:  Bashyolo task=detect mode=train model=yolov8n.pt data=data.yaml epochs=50 imgsz=640 device=mps
📊 Performance & ResultsThe model was trained on a custom dataset of FPV drone images.  Hardware: MacBook Pro M3 Pro (GPU Accelerated)  Best Accuracy: Reached an mAP50 of 0.82 within the first 10 epochs.  Inference Speed: Optimized for real-time monitoring.  🔍 InferenceTo run the detector on a video file or live stream:  Bashyolo task=detect mode=predict \
  model=runs/detect/train/weights/best.pt \
  source='path/to/fpv_footage.mp4' \
  show=True
📂 Project Structuredata.yaml: Configuration for dataset paths and classes.  runs/: Contains training logs, validation images, and model weights (best.pt).  requirements.txt: List of necessary Python packages (Ultralytics, OpenCV, etc.).  
