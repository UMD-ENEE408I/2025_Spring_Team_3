
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

class LineFollowerPID:
    def __init__(self):
        rospy.init_node("line_follower_pid", anonymous=True)
        
        # ROS Publishers and Subscribers
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.image_callback)
        self.cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)

        # Movement command
        self.twist = Twist()

        # Image width placeholder (gets set in callback)
        self.image_width = 0

        # PID parameters
        self.kp = 0.005
        self.ki = 0.0001
        self.kd = 0.001

        # PID state
        self.prev_error = 0
        self.integral = 0

    def image_callback(self, msg):
        # Convert ROS Image to OpenCV format
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        height, width, _ = frame.shape
        self.image_width = width

        # Crop the image to bottom 1/4 for better focus
        crop = frame[int(height * 0.75):, :]

        # Convert to grayscale and threshold to get white areas
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Get largest contour (assume it's the line)
            c = max(contours, key=cv2.contourArea)
            M = cv2.moments(c)

            if M["m00"] > 0:
                # Centroid of the white line
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                # Draw for visualization
                cv2.circle(crop, (cx, cy), 5, (0, 0, 255), -1)
                cv2.drawContours(crop, [c], -1, (0, 255, 0), 2)

                # PID control
                error = cx - (width // 2)
                self.integral += error
                derivative = error - self.prev_error
                correction = self.kp * error + self.ki * self.integral + self.kd * derivative
                self.prev_error = error

                # Update velocity command
                self.twist.linear.x = 0.15
                self.twist.angular.z = -correction
                self.cmd_vel_pub.publish(self.twist)

                rospy.loginfo(f"Centroid X: {cx}, Error: {error}, Correction: {correction:.4f}")
        else:
            # If line not found, stop or rotate
            rospy.logwarn("Line not found! Searching...")
            self.twist.linear.x = 0.0
            self.twist.angular.z = 0.3  # Rotate to search
            self.cmd_vel_pub.publish(self.twist)

        cv2.imshow("Line Follower View", crop)
        cv2.waitKey(1)

    def run(self):
        rospy.spin()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        LineFollowerPID().run()
    except rospy.ROSInterruptException:
        pass
