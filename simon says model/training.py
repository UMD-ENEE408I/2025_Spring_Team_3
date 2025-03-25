from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data="C:/Users/ebin5/ENEE408I_Files/2025_Spring_Team_3/simon says model/data.yaml", epochs=100, imgsz=640, name='custom_yolo_model', device='cpu')

model.save("simonsaysv1.pt")

 