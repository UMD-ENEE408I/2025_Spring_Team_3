#!/usr/bin/env python3
from ultralytics import YOLO
import rospy
from sensor_msgs.msg import CompressedImage
import cv2
from cv_bridge import CvBridge
import numpy as np

# Load your custom-trained YOLO model
model = YOLO("simonsaysv1.pt")

# ROS Node Configuration
subscriberNodeName = "camera_sensor_subscriber"
topicName = "video_topic/compressed"

# Initialize CV Bridge
bridgeObject = CvBridge()

def callbackFunction(message):
    rospy.loginfo("Received a video frame")

    try:
        # Convert from compressed image
        frame = bridgeObject.compressed_imgmsg_to_cv2(message, "bgr8")

        if frame is None:
            rospy.logwarn("Decoded frame is None")
            return

        # Display raw camera feed
        cv2.imshow("Camera Feed", frame)
        cv2.waitKey(1)  # Required to refresh display

        # Run YOLO model on the frame
        results = model(frame)

        # Get annotated frame with detected gestures
        annotated_frame = results[0].plot()

        # Display frame with detections
        cv2.imshow("Hand Signal Detection", annotated_frame)
        cv2.waitKey(1)  # Required to refresh display

    except Exception as e:
        rospy.logerr(f"Error processing frame: {e}")

if __name__ == "__main__":
    rospy.init_node(subscriberNodeName, anonymous=True)
    rospy.Subscriber(topicName, CompressedImage, callbackFunction, queue_size=1)
    
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down...")
    finally:
        cv2.destroyAllWindows()
