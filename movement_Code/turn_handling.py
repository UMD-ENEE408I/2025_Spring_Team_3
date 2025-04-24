#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
import time
from geometry_msgs.msg import Twist
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge

ENABLE_GUI = False

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

choice = input("Press 'p' to publish frames to Jetson, or 'n' to skip streaming: ").strip().lower()
send_frames = (choice == 'p')

rospy.init_node('line_follower_publisher', anonymous=True)
frame_pub = rospy.Publisher("video_topic/compressed", CompressedImage, queue_size=1)
cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)

cap = cv2.VideoCapture(0)
cap.set(3, 160)
cap.set(4, 120)
bridge = CvBridge()
pid = PID()

if not cap.isOpened():
    print("[ERROR] Cannot open camera.")
    exit(1)

def publish_frame(frame):
    if send_frames:
        resized = cv2.resize(frame, (640, 480))
        compressed = bridge.cv2_to_compressed_imgmsg(resized, dst_format="jpeg")
        frame_pub.publish(compressed)

# === Improved contrast and glare reduction ===
def process_roi(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Blur to reduce noise/glares
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive contrast: Otsu’s binarization
    _, binary = cv2.threshold(blurred, 180, 255, cv2.THRESH_BINARY)

    # Optional: Invert if needed (e.g., if line is black on white)
    # binary = cv2.bitwise_not(binary)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours, binary

MIN_CONTOUR_AREA = 200

while not rospy.is_shutdown():
    ret, frame = cap.read()
    if not ret:
        print("[WARN] Frame capture failed.")
        continue

    height, width, _ = frame.shape
    roi_bottom = frame[int(height * 0.75):, :]
    roi_top = frame[int(height * 0.5):int(height * 0.65), :]

    contours_bottom, binary_bottom = process_roi(roi_bottom)
    contours_top, binary_top = process_roi(roi_top)

    twist = Twist()

    valid_contours = [c for c in contours_bottom if cv2.contourArea(c) > MIN_CONTOUR_AREA]

    if valid_contours:
        largest = max(valid_contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            error = cx - width // 2
            correction = pid.compute(error)

            if cx < width * 0.2:
                twist.linear.x = -0.01
                twist.angular.z = -0.5
                rospy.loginfo("Sharp RIGHT turn (flipped)")
            elif cx > width * 0.8:
                twist.linear.x = -0.01
                twist.angular.z = 0.5
                rospy.loginfo("Sharp LEFT turn (flipped)")
            else:
                twist.linear.x = -0.03
                twist.angular.z = -correction * 0.5
                rospy.loginfo(f"[INFO] Tracking (flipped) | Centroid: {cx}, Correction: {correction:.3f}")
                cv2.drawContours(roi_bottom, [largest], -1, (0, 255, 0), 2)
                cv2.circle(roi_bottom, (cx, int(M["m01"] / M["m00"])), 5, (0, 0, 255), -1)

        cmd_pub.publish(twist)

    elif contours_top:
        twist.linear.x = 0.0
        twist.angular.z = -0.4
        rospy.logwarn("Line lost below, found above → turning RIGHT to reacquire")
        cmd_pub.publish(twist)

    else:
        rospy.logwarn("Line lost in both ROIs → assuming 90° RIGHT turn")
        twist.linear.x = 0.0
        twist.angular.z = -0.4

        turn_start = rospy.Time.now().to_sec()
        turn_duration = 0.8
        rate = rospy.Rate(15)
        while rospy.Time.now().to_sec() - turn_start < turn_duration and not rospy.is_shutdown():
            ret_turn, frame_turn = cap.read()
            if ret_turn:
                cmd_pub.publish(twist)
                publish_frame(frame_turn)
                rate.sleep()

        twist.angular.z = 0.0
        cmd_pub.publish(twist)
        rospy.sleep(0.2)

    # Always publish current frame
    publish_frame(frame)

    if ENABLE_GUI:
        try:
            cv2.imshow("Bottom ROI", binary_bottom)
            cv2.imshow("Top ROI", binary_top)
            cv2.imshow("Camera Feed", frame)
        except cv2.error as e:
            rospy.logwarn(f"[WARN] GUI display failed: {e}")

    if ENABLE_GUI and cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
if ENABLE_GUI:
    cv2.destroyAllWindows()
cmd_pub.publish(Twist())
