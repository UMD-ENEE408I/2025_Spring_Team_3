#!/usr/bin/env python3

# THIS CODE DID PRETTY WELL SAVE IT
import rospy
import cv2
import numpy as np
import time
from geometry_msgs.msg import Twist
from sensor_msgs.msg import CompressedImage
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
from cv_bridge import CvBridge

ENABLE_GUI = False

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

# === Globals for Odometry ===
current_yaw = 0.0
odom_received = False
turning = False

def odom_callback(msg):
    global current_yaw, odom_received
    orientation_q = msg.pose.pose.orientation
    _, _, yaw = euler_from_quaternion([
        orientation_q.x,
        orientation_q.y,
        orientation_q.z,
        orientation_q.w
    ])
    current_yaw = yaw
    odom_received = True

def normalize_angle(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))

def turn_left(angle_degrees=90):
    global current_yaw, odom_received
    while not odom_received and not rospy.is_shutdown():
        rospy.sleep(0.1)

    initial_yaw = current_yaw
    target_angle = np.deg2rad(angle_degrees)
    target_yaw = normalize_angle(initial_yaw + target_angle)
    base_speed = 0.5

    twist = Twist()
    twist.linear.x = 0.0
    rate = rospy.Rate(15)

    while not rospy.is_shutdown():
        error = normalize_angle(target_yaw - current_yaw)

        # Slow down if within 30°
        twist.angular.z = base_speed if abs(error) > np.deg2rad(30) else base_speed * 0.5

        # Break if we're within 2°
        if abs(error) < np.deg2rad(2):
            break

        cmd_pub.publish(twist)
        ret, frame = cap.read()
        if ret:
            publish_frame(frame)
        rate.sleep()

    cmd_pub.publish(Twist())
    rospy.sleep(0.5)

def turn_right(angle_degrees=90):
    global current_yaw, odom_received
    while not odom_received and not rospy.is_shutdown():
        rospy.sleep(0.1)

    initial_yaw = current_yaw
    target_angle = np.deg2rad(angle_degrees)
    target_yaw = normalize_angle(initial_yaw - target_angle)
    base_speed = -0.5

    twist = Twist()
    twist.linear.x = 0.0
    rate = rospy.Rate(15)

    while not rospy.is_shutdown():
        error = normalize_angle(target_yaw - current_yaw)

        # Slow down if within 30°
        twist.angular.z = base_speed if abs(error) > np.deg2rad(30) else base_speed * 0.5

        # Break if we're within 2°
        if abs(error) < np.deg2rad(2):
            break

        cmd_pub.publish(twist)
        ret, frame = cap.read()
        if ret:
            publish_frame(frame)
        rate.sleep()

    cmd_pub.publish(Twist())
    rospy.sleep(0.5)

def look_for_line(direction='right', angle_degrees=90):
    if direction == 'right':
        turn_right(angle_degrees)
    else:
        turn_left(angle_degrees)
        
    rospy.sleep(0.5)
    ret, frame = cap.read()
    if not ret:
        return False
    height, width, _ = frame.shape
    roi = frame[int(height * 0.75):, :]
    _, decision = process_roi_and_decide(roi)
    return decision != "lost"

# === Initialize ROS ===
choice = input("Press 'p' to publish frames to Jetson, or 'n' to skip streaming: ").strip().lower()
send_frames = (choice == 'p')

rospy.init_node('line_follower_publisher', anonymous=True)
frame_pub = rospy.Publisher("video_topic/compressed", CompressedImage, queue_size=1)
cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
rospy.Subscriber("/odom", Odometry, odom_callback)

cap = cv2.VideoCapture(0)
cap.set(3, 160)
cap.set(4, 120)
bridge = CvBridge()
pid = PID()

def publish_frame(frame):
    if send_frames:
        # Create mask from frame (detect white regions like process_roi)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

        # Find contours from the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw the contours on the frame (in green)
        cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)

        # Resize for transmission
        resized = cv2.resize(frame, (640, 480))
        compressed = bridge.cv2_to_compressed_imgmsg(resized, dst_format="jpeg")
        frame_pub.publish(compressed)


# === Initial delay while showing and publishing frames ===
rospy.loginfo("Warming up camera and waiting 5 seconds...")
start_time = rospy.Time.now().to_sec()
while rospy.Time.now().to_sec() - start_time < 5 and not rospy.is_shutdown():
    ret, frame = cap.read()
    if ret:
        publish_frame(frame)
        if ENABLE_GUI:
            try:
                cv2.imshow("Startup Camera View", frame)
                cv2.waitKey(1)
            except cv2.error as e:
                rospy.logwarn(f"[WARN] GUI display failed: {e}")
    rospy.sleep(0.05)

if not cap.isOpened():
    print("[ERROR] Cannot open camera.")
    exit(1)

def process_roi_and_decide(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    height, width = binary.shape
    left_zone = binary[:, :width // 3]
    center_zone = binary[:, width // 3: 2 * width // 3]
    right_zone = binary[:, 2 * width // 3:]

    left_sum = np.sum(left_zone == 255)
    center_sum = np.sum(center_zone == 255)
    right_sum = np.sum(right_zone == 255)

    decision = "lost"
    if center_sum > max(left_sum, right_sum) * 0.7:
        decision = "straight"
    elif left_sum > right_sum:
        decision = "left"
    elif right_sum > left_sum:
        decision = "right"

    return binary, decision

# === Main Loop ===
MIN_CONTOUR_AREA = 200

while not rospy.is_shutdown():
    ret, frame = cap.read()
    if not ret:
        print("[WARN] Frame capture failed.")
        continue

    height, width, _ = frame.shape
    roi_bottom = frame[int(height * 0.75):, :]
    roi_top = frame[int(height * 0.5):int(height * 0.65), :]

    binary_bottom, decision_bottom = process_roi_and_decide(roi_bottom)
    binary_top, decision_top = process_roi_and_decide(roi_top)

    twist = Twist()

    if decision_bottom == "straight" and not turning:
        twist.linear.x = -0.03
        twist.angular.z = 0.0
        cmd_pub.publish(twist)
        rospy.loginfo("[INFO] Following line straight.")

    elif decision_bottom == "left" and not turning:
        twist.linear.x = 0.02
        twist.angular.z = 0.5
        cmd_pub.publish(twist)
        rospy.loginfo("[INFO] Curve detected → turning LEFT.")

    elif decision_bottom == "right" and not turning:
        twist.linear.x = 0.02
        twist.angular.z = -0.5
        cmd_pub.publish(twist)
        rospy.loginfo("[INFO] Curve detected → turning RIGHT.")

    elif decision_top != "lost" and not turning:
        twist.linear.x = 0.0
        twist.angular.z = -0.4
        cmd_pub.publish(twist)
        rospy.logwarn("Line lost below, found above → turning RIGHT to reacquire")

    elif not turning:
        turning = True
        rospy.logwarn("Line lost in both ROIs → scanning...")
        found = look_for_line('right', 45)
        if found:
            rospy.loginfo("[INFO] Line found after RIGHT scan")
            turning = False
            continue

        found = look_for_line('left', 180)
        if found:
            rospy.loginfo("[INFO] Line found after LEFT 180 scan")
            turning = False
            continue

        rospy.logwarn("No line found after scanning → stopping robot.")
        cmd_pub.publish(Twist())
        break

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
