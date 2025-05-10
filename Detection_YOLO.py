#!/usr/bin/env python3

from ultralytics import YOLO
import rospy
from sensor_msgs.msg import CompressedImage
import cv2
from cv_bridge import CvBridge
import time
import numpy as np
import threading

DISPLAY_FPS = True
SKIP_FRAMES = True
MAX_INFERENCE_SIZE = (320, 240)  
CONFIDENCE_THRESHOLD = 0.5       

model = YOLO("simonsaysv1.pt")
model.conf = CONFIDENCE_THRESHOLD  
model.iou = 0.45  
model.agnostic = False  
model.multi_label = False  
model.max_det = 10 

subscriberNodeName = "camera_sensor_subscriber"
topicName = "video_topic/compressed"

bridgeObject = CvBridge()

last_frame_time = None
processing_frame = False
skip_count = 0
latest_frame = None
is_processing = False
fps_history = []
latency_history = []

def update_display_stats(frame, fps, latency):
    """Add FPS and latency information to the frame"""
    if DISPLAY_FPS:
        
        cv2.rectangle(frame, (0, 0), (250, 60), (0, 0, 0), -1)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Latency: {latency*1000:.0f} ms", (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame

def callbackFunction(message):
    global latest_frame, is_processing, last_frame_time, skip_count
    
    try:
        
        frame_time = time.time()
        last_frame_time = frame_time
        
        
        frame = bridgeObject.compressed_imgmsg_to_cv2(message, "bgr8")
        if frame is None:
            rospy.logwarn("Decoded frame is None")
            return
            
        
        latest_frame = frame
        
        
        if is_processing and SKIP_FRAMES:
            skip_count += 1
            return
            
        
        if not is_processing:
            is_processing = True
            thread = threading.Thread(target=process_frame, args=(frame, frame_time))
            thread.daemon = True
            thread.start()
            
    except Exception as e:
        rospy.logerr(f"Error converting frame: {e}")

def process_frame(frame, frame_time):
    global is_processing, fps_history, latency_history, skip_count
    
    start_time = time.time()
    
    try:
        # Resize for faster inference
        resized_frame = cv2.resize(frame, MAX_INFERENCE_SIZE)
        
        # Run YOLO model with hardware acceleration
        results = model(resized_frame, verbose=False)
        
        # Annotate original frame for better display quality
        annotated_frame = results[0].plot()
        
        # Calculate processing performance
        end_time = time.time()
        processing_time = end_time - start_time
        latency = end_time - frame_time
        
        # Keep track of performance metrics
        fps_history.append(1.0 / processing_time)
        latency_history.append(latency)
        
        # Only keep recent history for smoother display
        if len(fps_history) > 30:
            fps_history.pop(0)
        if len(latency_history) > 30:
            latency_history.pop(0)
            
        avg_fps = np.mean(fps_history)
        avg_latency = np.mean(latency_history)
        
        # Add performance stats to frame
        annotated_frame = update_display_stats(annotated_frame, avg_fps, avg_latency)
        
        # If we've skipped frames, log it
        if skip_count > 0:
            rospy.loginfo(f"Skipped {skip_count} frames to keep up")
            skip_count = 0

        cv2.namedWindow("Hand Signal Detection", cv2.WINDOW_NORMAL)
        # Show detection
        cv2.imshow("Hand Signal Detection", annotated_frame)
        cv2.waitKey(1)  # Required to refresh display
        
    except Exception as e:
        rospy.logerr(f"Error in frame processing: {e}")
    finally:
        # Check if there's a newer frame to process immediately
        global latest_frame, last_frame_time
        if latest_frame is not None and last_frame_time is not None and frame_time < last_frame_time:
            # Process the latest frame immediately
            next_frame = latest_frame
            next_time = last_frame_time
            latest_frame = None
            process_frame(next_frame, next_time)
        else:
            # Mark as not processing
            is_processing = False

def check_for_stuck_frames():
    """Periodically check if frames are being processed and unstick if needed"""
    global is_processing
    
    rate = rospy.Rate(1)  # Check every 1 second
    while not rospy.is_shutdown():
        if is_processing and time.time() - last_frame_time > 3.0:
            rospy.logwarn("Processing appears stuck, resetting state")
            is_processing = False
        rate.sleep()

if __name__ == "__main__":
    rospy.init_node(subscriberNodeName, anonymous=True)
    
    
    rospy.Subscriber(topicName, CompressedImage, callbackFunction, queue_size=1, 
                    buff_size=2**24)  
    
    watchdog_thread = threading.Thread(target=check_for_stuck_frames)
    watchdog_thread.daemon = True
    watchdog_thread.start()
    
    try:

        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down...")
    finally:
        cv2.destroyAllWindows()
