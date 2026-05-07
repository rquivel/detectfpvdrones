from ultralytics import YOLO

model = YOLO('./runs/detect/train_v2/weights/best.pt')

results = model.predict(source='./dataset/images/train/image3.png', conf=0.25)

results[0].show()