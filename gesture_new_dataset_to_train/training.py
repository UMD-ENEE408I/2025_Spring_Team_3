from ultralytics import YOLO

# # Load a model
# model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)

# # Train the model
# results = model.train(data="C:/Users/ebin5/ENEE408I_Files/2025_Spring_Team_3/gesture_new_dataset_to_train/data.yaml", epochs=20, imgsz=640, name='custom_yolo_model', device='cuda')

def train_model():
    model = YOLO("yolo11n.pt")  # Load the YOLO model

    # Train the model on GPU
    results = model.train(
        data="C:/Users/ebin5/ENEE408I_Files/2025_Spring_Team_3/gesture_new_dataset_to_train/data.yaml",
        epochs=100, imgsz=640,
        name='custom_yolo_model',
        device='cuda'  # Run on GPU
    )

    print("Training complete. Check the weights folder for the trained model.")
    model.save("simonsaysv1_100_epochs.pt")

if __name__ == '__main__':
    train_model()