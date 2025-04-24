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

def detect_line(binary_image, min_area=200):
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) > min_area:
            return True
    return False

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

def publish_frame(frame):
    if send_frames:
        resized = cv2.resize(frame, (640, 480))
        compressed = bridge.cv2_to_compressed_imgmsg(resized, dst_format="jpeg")
        frame_pub.publish(compressed)

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
            twist.linear.x = -0.03
            twist.angular.z = -correction * 0.5
            cv2.drawContours(roi, [largest], -1, (0, 255, 0), 2)
            cv2.circle(roi, (cx, cy), 5, (0, 0, 255), -1)
    else:
        rospy.logwarn("[WARN] No line detected. Searching for line...")
        twist.linear.x = 0.0
        twist.angular.z = -0.4
        cmd_pub.publish(twist)

        turn_start = rospy.Time.now().to_sec()
        while rospy.Time.now().to_sec() - turn_start < 0.8 and not rospy.is_shutdown():
            ret_turn, frame_turn = cap.read()
            if ret_turn:
                height, width, _ = frame_turn.shape
                roi = frame_turn[int(height * 0.75):, :]
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
                publish_frame(frame_turn)
                if detect_line(binary):
                    rospy.loginfo("[INFO] Line found to the right. Resuming forward motion.")
                    break
                twist.angular.z = -0.4
                cmd_pub.publish(twist)
                rate.sleep()
        else:
            # Right side failed. Try left turn (~180° total from original heading)
            twist.angular.z = 0.4
            turn_start = rospy.Time.now().to_sec()
            while rospy.Time.now().to_sec() - turn_start < 1.6 and not rospy.is_shutdown():
                ret_turn, frame_turn = cap.read()
                if ret_turn:
                    height, width, _ = frame_turn.shape
                    roi = frame_turn[int(height * 0.75):, :]
                    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
                    publish_frame(frame_turn)
                    if detect_line(binary):
                        rospy.loginfo("[INFO] Line found to the left. Resuming forward motion.")
                        break
                    twist.angular.z = 0.4
                    cmd_pub.publish(twist)
                    rate.sleep()
            else:
                # No line found in either direction
                rospy.logwarn("[WARN] No line in either direction. Stopping.")
                twist.linear.x = 0.0
                twist.angular.z = 0.0

    cmd_pub.publish(twist)
    publish_frame(frame)

    if ENABLE_GUI:
        try:
            cv2.imshow("ROI", roi)
            cv2.imshow("Binary", binary)
            cv2.imshow("Camera", frame)
        except cv2.error as e:
            rospy.logwarn(f"[WARN] GUI error: {e}")

    if ENABLE_GUI and cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
if ENABLE_GUI:
    cv2.destroyAllWindows()
cmd_pub.publish(Twist())
