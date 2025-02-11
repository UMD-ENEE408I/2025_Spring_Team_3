from ultralytics import YOLO
import cv2

# Load the YOLOv8 model (pre-trained on COCO dataset)
model = YOLO("yolov8n.pt")  

# Open webcam (0 = default webcam, 1 = external camera/HDMI device)
cap = cv2.VideoCapture(1)  # Change to 1 if your laptop camera is on index 1

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()  # Read a frame from the webcam
    if not ret:
        break

    # Run YOLO on the frame
    results = model(frame)

    # Get annotated frame with detected objects
    annotated_frame = results[0].plot()

    # Display the frame
    cv2.imshow("YOLO Webcam Feed", annotated_frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
