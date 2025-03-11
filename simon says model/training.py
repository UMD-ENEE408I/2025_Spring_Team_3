from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data="C:/Users/gough/2025_Spring_Team_3/2025_Spring_Lab_3-main/data.yaml", epochs=100, imgsz=640, name='custom_yolo_model', device='cpu')

model.save("simonsaysv1.pt")

