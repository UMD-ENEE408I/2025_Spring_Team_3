#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
import time
from geometry_msgs.msg import Twist
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge

ENABLE_GUI = False  # Enable GUI only if not headless

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
rate = rospy.Rate(10)

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

    bottom_roi = frame[int(height * 0.75):, :]
    mid_roi = frame[int(height * 0.5):int(height * 0.75), :]

    def process_roi(roi):
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours, binary

    bottom_contours, bottom_bin = process_roi(bottom_roi)
    mid_contours, mid_bin = process_roi(mid_roi)

    twist = Twist()
    turning = False

    # === Preemptive turn if mid sees line but bottom does not ===
    if not bottom_contours and mid_contours:
        largest = max(mid_contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            if cx < width // 2:
                rospy.loginfo("[TURNING] Preemptive LEFT turn")
                twist.linear.x = -0.02
                twist.angular.z = 0.5
            else:
                rospy.loginfo("[TURNING] Preemptive RIGHT turn")
                twist.linear.x = -0.02
                twist.angular.z = -0.5
            turning = True

    # === Normal PID tracking ===
    elif bottom_contours:
        largest = max(bottom_contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            error = cx - width // 2
            correction = pid.compute(error)

            twist.linear.x = -0.03
            twist.angular.z = -correction * 0.5
            rospy.loginfo(f"[PID] cx={cx}, error={error}, correction={correction:.3f}")

            if ENABLE_GUI:
                cv2.drawContours(bottom_roi, [largest], -1, (0, 255, 0), 2)
                cv2.circle(bottom_roi, (cx, int(bottom_roi.shape[0]/2)), 5, (0, 0, 255), -1)
    else:
        rospy.logwarn("[STOP] No line in bottom or mid ROI")
        twist.linear.x = 0.0
        twist.angular.z = 0.0

    cmd_pub.publish(twist)

    if send_frames:
        frame = cv2.resize(frame, (640, 480))
        compressedImage = bridge.cv2_to_compressed_imgmsg(frame, dst_format="jpeg")
        frame_pub.publish(compressedImage)

    if ENABLE_GUI:
        try:
            cv2.imshow("Bottom ROI", bottom_roi)
            cv2.imshow("Mid ROI", mid_roi)
            cv2.imshow("Bottom Mask", bottom_bin)
            cv2.imshow("Mid Mask", mid_bin)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except cv2.error as e:
            rospy.logwarn(f"[WARN] GUI error: {e}")

cap.release()
if ENABLE_GUI:
    cv2.destroyAllWindows()
cmd_pub.publish(Twist())
