from ultralytics import YOLO

# Load a pre-trained YOLO model
model = YOLO("yolov8n.pt")  # Start with a pre-trained model

# Train on your custom dataset
model.train(data=".\Turtlebots.yolov8\data.yaml", epochs=100, imgsz=640)
