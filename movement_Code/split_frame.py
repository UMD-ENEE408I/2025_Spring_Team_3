#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
import time
from geometry_msgs.msg import Twist
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge

ENABLE_GUI = True

class PID:
    def __init__(self, kp=0.001, ki=0.00001, kd=0.0001):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0
        self.integral = 0
        self.last_time = time.time()

    def compute(self, error):
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        self.prev_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative

def get_centroid(binary_img):
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return cx, cy, largest
    return None, None, None

def preprocess_roi(roi):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 40, 255])
    mask_hsv = cv2.inRange(hsv, lower_white, upper_white)

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    mask_adapt = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

    combined = cv2.bitwise_and(mask_hsv, mask_adapt)
    cleaned = cv2.morphologyEx(combined, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return cleaned

def initialize_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(
            'nvarguscamerasrc ! video/x-raw(memory:NVMM), width=640, height=480, framerate=30/1 ! '
            'nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink',
            cv2.CAP_GSTREAMER
        )
    return cap

# === ROS and Camera Init ===
choice = input("Press 'p' to publish frames to Jetson, or 'n' to skip streaming: ").strip().lower()
send_frames = (choice == 'p')

rospy.init_node('line_follower_publisher', anonymous=True)
frame_pub = rospy.Publisher("video_topic/compressed", CompressedImage, queue_size=1)
cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
rate = rospy.Rate(10)

cap = initialize_camera()
cap.set(3, 160)
cap.set(4, 120)
bridge = CvBridge()
pid = PID()

if not cap.isOpened():
    print("[ERROR] Cannot open camera.")
    exit(1)

# Check GUI availability
try:
    test = np.zeros((10, 10), dtype=np.uint8)
    cv2.imshow("Test", test)
    cv2.waitKey(1)
    cv2.destroyAllWindows()
except cv2.error:
    ENABLE_GUI = False
    print("[INFO] GUI not available — running in headless mode.")

# === Main Loop ===
while not rospy.is_shutdown():
    ret, frame = cap.read()
    if not ret:
        print("[WARN] Frame capture failed.")
        continue

    height, width, _ = frame.shape
    twist = Twist()

    # Define ROIs
    bottom_roi = frame[int(height * 0.65):, :]
    top_roi = frame[int(height * 0.3):int(height * 0.5), :]

    # Preprocess ROIs
    binary_bottom = preprocess_roi(bottom_roi)
    cx_b, cy_b, _ = get_centroid(binary_bottom)

    binary_top = preprocess_roi(top_roi)
    cx_t, _, _ = get_centroid(binary_top)

    # === Control Logic ===
    if cx_b is None and cx_t is not None and cx_t > 0.7 * width:
        # Preemptive right turn
        twist.linear.x = -0.02
        twist.angular.z = -0.3
        mode_text = "Right Turn (Top ROI)"
        mode_color = (0, 255, 255)
    elif cx_b is not None:
        # PID Line Following
        error = cx_b - width // 2
        correction = pid.compute(error)
        twist.linear.x = -0.03
        twist.angular.z = -correction * 0.5
        mode_text = "PID Line Following"
        mode_color = (0, 255, 0)
    else:
        # No line visible — stop movement
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        mode_text = "No Line Visible"
        mode_color = (100, 100, 100)

    cmd_pub.publish(twist)

    # === GUI Debug Overlay ===
    if ENABLE_GUI:
        debug_frame = frame.copy()
        b_y = int(height * 0.65)
        t_y = int(height * 0.3)

        # Bottom ROI box
        cv2.rectangle(debug_frame, (0, b_y), (width, height), (255, 255, 255), 1)
        if cx_b is not None:
            cv2.putText(debug_frame, "Bottom ROI: Line Detected", (5, b_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.circle(debug_frame, (cx_b, b_y + cy_b), 5, (0, 0, 255), -1)
        else:
            cv2.putText(debug_frame, "Bottom ROI: No Line", (5, b_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Top ROI box
        cv2.rectangle(debug_frame, (0, t_y), (width, int(height * 0.5)), (180, 180, 255), 1)
        if cx_t is not None:
            cv2.putText(debug_frame, "Top ROI: Line Detected", (5, t_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)
            cv2.circle(debug_frame, (cx_t, t_y + 15), 5, (0, 165, 255), -1)
        else:
            cv2.putText(debug_frame, "Top ROI: No Line", (5, t_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 1)

        # Display mode and velocity
        cv2.putText(debug_frame, f"MODE: {mode_text}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, mode_color, 2)
        cv2.putText(debug_frame, f"Vel: x={twist.linear.x:.2f}, z={twist.angular.z:.2f}", (5, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Show windows
        cv2.imshow("Debug View", debug_frame)
        cv2.imshow("Bottom ROI Mask", binary_bottom)
        cv2.imshow("Top ROI Mask", binary_top)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Optional streaming
    if send_frames:
        frame_out = cv2.resize(frame, (640, 480))
        compressedImage = bridge.cv2_to_compressed_imgmsg(frame_out, dst_format="jpeg")
        frame_pub.publish(compressedImage)

cap.release()
if ENABLE_GUI:
    cv2.destroyAllWindows()
cmd_pub.publish(Twist())