#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
import time
from geometry_msgs.msg import Twist
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge

# === Enable or disable GUI display ===
ENABLE_GUI = False  # Set to True ONLY if GUI is supported (not over SSH/headless Jetson)

# === PID Controller ===
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

# === Prompt User Input ===
choice = input("Press 'p' to publish frames to Jetson, or 'n' to skip streaming: ").strip().lower()
send_frames = (choice == 'p')

# === ROS Init ===
rospy.init_node('line_follower_publisher', anonymous=True)
frame_pub = rospy.Publisher("video_topic/compressed", CompressedImage, queue_size=1)
cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
rate = rospy.Rate(10)

# === Camera and Tools Init ===
cap = cv2.VideoCapture(0)
cap.set(3, 160)
cap.set(4, 120)
bridge = CvBridge()
pid = PID()

if not cap.isOpened():
    print("[ERROR] Cannot open camera.")
    exit(1)

while not rospy.is_shutdown():
    ret, frame = cap.read()
    if not ret:
        print("[WARN] Frame capture failed.")
        continue

    height, width, _ = frame.shape
    roi = frame[int(height * 0.75):, :]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    twist = Twist()

    if contours:
        largest = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            error = cx - width // 2
            correction = pid.compute(error)

            rospy.loginfo(f"[INFO] Line Detected | Centroid: {cx}, Error: {error}, Correction: {correction:.3f}")
            
            # === Movement Fixes ===
            # This variable controls speed.
            twist.linear.x = -0.03               # Correct forward direction (camera-facing)
            twist.angular.z = -correction * 0.5   # Correct turning direction
            cv2.drawContours(roi, [largest], -1, (0, 255, 0), 2)
            cv2.circle(roi, (cx, cy), 5, (0, 0, 255), -1)
    else:
        rospy.logwarn("[WARN] No line detected. Stopping.")
        twist.linear.x = 0.0
        twist.angular.z = 0.0

    cmd_pub.publish(twist)

    # === Frame Publishing ===
    if send_frames:
        frame = cv2.resize(frame, (640, 480))
        compressedImage = bridge.cv2_to_compressed_imgmsg(frame, dst_format="jpeg")
        frame_pub.publish(compressedImage)

        if ENABLE_GUI:
            try:
                cv2.imshow("Publishing Camera Feed", frame)
            except cv2.error as e:
                rospy.logwarn(f"[WARN] GUI display failed: {e}")
    elif ENABLE_GUI:
        try:
            cv2.imshow("Line Detection View", roi)
            cv2.imshow("Binary Mask", binary)
        except cv2.error as e:
            rospy.logwarn(f"[WARN] GUI display failed: {e}")

    if ENABLE_GUI and cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
if ENABLE_GUI:
    cv2.destroyAllWindows()
cmd_pub.publish(Twist())  # Stop robot on exit
