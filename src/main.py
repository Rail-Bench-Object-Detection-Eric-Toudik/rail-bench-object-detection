from ultralytics import YOLO
model = YOLO('yolov11n.pt')
model.train(data='data/yolo_config.yaml', epochs=100, imgsz=640)