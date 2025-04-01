#!/usr/bin/env python3
from ultralytics import YOLO
import rospy
from sensor_msgs.msg import CompressedImage
import cv2
from cv_bridge import CvBridge
from threading import Thread
from queue import Queue

# Load your custom-trained YOLO model
model = YOLO("simonsaysv1.pt")

# ROS Node Configuration
subscriberNodeName = "camera_sensor_subscriber"
topicName = "video_topic/compressed"

# Initialize CV Bridge
bridgeObject = CvBridge()

# Queue for threading
frame_queue = Queue(maxsize=2)

def callbackFunction(message):
    try:
        # Convert from compressed image
        frame = bridgeObject.compressed_imgmsg_to_cv2(message, "bgr8")
        if frame is None:
            rospy.logwarn("Decoded frame is None")
            return

        # Resize for faster inference (you can tweak the size)
        resized_frame = cv2.resize(frame, (640, 480))

        # Push to queue for processing if not full
        if not frame_queue.full():
            frame_queue.put(resized_frame)

    except Exception as e:
        rospy.logerr(f"Error converting frame: {e}")

def process_frames():
    while not rospy.is_shutdown():
        if not frame_queue.empty():
            frame = frame_queue.get()

            # Run YOLO model
            results = model(frame)

            # Annotate frame
            annotated_frame = results[0].plot()

            # Show detection
            cv2.imshow("Hand Signal Detection", annotated_frame)
            cv2.waitKey(1)  # Required to refresh display

if __name__ == "__main__":
    rospy.init_node(subscriberNodeName, anonymous=True)
    rospy.Subscriber(topicName, CompressedImage, callbackFunction, queue_size=1)

    # Start separate thread for inference
    Thread(target=process_frames, daemon=True).start()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down...")
    finally:
        cv2.destroyAllWindows()
