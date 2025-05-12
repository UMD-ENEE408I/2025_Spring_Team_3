#!/usr/bin/env python3

import rospy
import cv2
import re
import threading
import queue
import torch
import math
from ultralytics import YOLO
from sensor_msgs.msg import CompressedImage
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
from tf.transformations import euler_from_quaternion
from google.cloud import speech
from google.oauth2 import service_account
import pyaudio

# === Constants ===
RATE = 16000
CHUNK = int(RATE / 10)
INCH_TO_M = 0.0254
DEFAULT_DISTANCE_IN = 6    # inches
TURN_ANGLE_DEG = 90        # degrees
ANGULAR_SPEED = 0.5        # rad/s
LINEAR_SPEED = 0.2         # m/s

# === YOLO Setup ===
model = YOLO("simonsaysv1.pt")
model.to("cuda")
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Model on device:", next(model.model.parameters()).device)

# === ROS + CV Bridge Setup ===
subscriber_node_name = "simon_yolo_odometry_node"
topic_name = "video_topic/compressed"
bridge = CvBridge()
frame_queue = queue.Queue(maxsize=2)

# === Odometry‑based Mover ===
class OdometricTurtleMover:
    def __init__(self, linear_speed=LINEAR_SPEED, angular_speed=ANGULAR_SPEED):
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        rospy.Subscriber('/odom', Odometry, self._odom_cb)
        self.linear_speed = linear_speed
        self.angular_speed = angular_speed
        self.start_x = None
        self.start_y = None
        self.current_x = None
        self.current_y = None
        self.start_yaw = None
        self.current_yaw = None

    def _odom_cb(self, msg):
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        if self.start_x is None:
            self.start_x = pos.x
            self.start_y = pos.y
        self.current_x = pos.x
        self.current_y = pos.y
        q = [ori.x, ori.y, ori.z, ori.w]
        _, _, yaw = euler_from_quaternion(q)
        if self.start_yaw is None:
            self.start_yaw = yaw
        self.current_yaw = yaw

    def move_distance(self, inches, forward=True):
        rate = rospy.Rate(20)
        while self.current_x is None and not rospy.is_shutdown():
            rate.sleep()
        distance_m = inches * INCH_TO_M
        start_x, start_y = self.current_x, self.current_y
        twist = Twist()
        twist.linear.x = self.linear_speed * (1 if forward else -1)
        rospy.loginfo(f"🚀 Moving {'forward' if forward else 'backward'} {inches} in (~{distance_m:.2f} m)")
        while not rospy.is_shutdown():
            dx = self.current_x - start_x
            dy = self.current_y - start_y
            if math.hypot(dx, dy) >= distance_m:
                break
            self.cmd_pub.publish(twist)
            rate.sleep()
        self.cmd_pub.publish(Twist())
        rospy.loginfo("🛑 Movement complete")

    def move_forward(self, inches=DEFAULT_DISTANCE_IN):
        self.move_distance(inches, forward=True)

    def move_backward(self, inches=DEFAULT_DISTANCE_IN):
        self.move_distance(inches, forward=False)

    def turn_degrees(self, angle_deg):
        rate = rospy.Rate(20)
        while self.current_yaw is None and not rospy.is_shutdown():
            rate.sleep()
        start = self.current_yaw
        delta = math.radians(angle_deg)
        goal = (start + delta + math.pi) % (2*math.pi) - math.pi
        twist = Twist()
        twist.angular.z = self.angular_speed * (1 if delta > 0 else -1)
        rospy.loginfo(f"🔄 Turning {angle_deg}° to yaw {goal:.2f} rad")
        while not rospy.is_shutdown():
            err = (goal - self.current_yaw + math.pi) % (2*math.pi) - math.pi
            if abs(err) < math.radians(2):
                break
            self.cmd_pub.publish(twist)
            rate.sleep()
        self.cmd_pub.publish(Twist())
        rospy.loginfo("🛑 Turn complete")

# instantiate mover placeholder
mover = None

# === Shared State ===
simon_mode = False
state_lock = threading.Lock()

# === Movement Wrappers ===
def move_forward():
    mover.move_forward()

def move_backward():
    mover.move_backward()

def turn_left():
    mover.turn_degrees(TURN_ANGLE_DEG)

def turn_right():
    mover.turn_degrees(-TURN_ANGLE_DEG)

def stop_movement():
    mover.move_forward(0)

# === Image Callback: enqueue frames ===
def image_callback(msg):
    try:
        frame = bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
        if frame is None:
            rospy.logwarn("⚠️ Frame conversion returned None")
            return
        resized = cv2.resize(frame, (640, 480))
        if not frame_queue.full():
            frame_queue.put(resized)
    except Exception as e:
        rospy.logerr(f"❌ Error converting frame: {e}")

# === Processing Thread: run YOLO & trigger moves in Simon mode ===
def process_frames():
    global simon_mode
    while not rospy.is_shutdown():
        try:
            frame = frame_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        with state_lock:
            sm = simon_mode
        res = model(frame, verbose=sm)
        results = res[0]
        annotated = results.plot()
        if sm:
            with state_lock:
                for box in results.boxes:
                    label = model.names[int(box.cls)]
                    if label == "forward":
                        move_forward()
                        simon_mode = False
                        break
                    elif label == "backward":
                        move_backward()
                        simon_mode = False
                        break
                    elif label == "left":
                        turn_left()
                        simon_mode = False
                        break
                    elif label == "right":
                        turn_right()
                        simon_mode = False
                        break
                    elif label == "stop":
                        stop_movement()
                        simon_mode = False
                        break
        cv2.imshow("YOLOv11 Gesture", annotated)
        cv2.waitKey(1)

# === Audio → “Simon says” detection ===
class MicrophoneStream:
    def __init__(self, rate=RATE, chunk=CHUNK):
        self._rate = rate
        self._chunk = chunk
        self._buff = queue.Queue()
        self.closed = True

    def __enter__(self):
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self._rate,
            input=True,
            frames_per_buffer=self._chunk,
            stream_callback=self._fill_buffer,
        )
        self.closed = False
        rospy.loginfo("🎤 Audio thread started")
        return self

    def __exit__(self, exc_type, exc, exc_tb):
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self):
        while not self.closed:
            chunk = self._buff.get()
            if chunk is None:
                return
            data = [chunk]
            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break
            yield b"".join(data)

def listen_print_loop(responses):
    global simon_mode
    for response in responses:
        if not response.results or not response.results[0].alternatives:
            continue
        result = response.results[0]
        transcript = result.alternatives[0].transcript
        if result.is_final:
            rospy.loginfo(f"🗣️ Heard: {transcript}")
            if re.search(r"\bsimon says\b", transcript, re.I):
                with state_lock:
                    simon_mode = True
                rospy.loginfo("🗣️ Simon says! Next gesture will move.")

def audio_thread_fn():
    creds = service_account.Credentials.from_service_account_file("key.json")
    client = speech.SpeechClient(credentials=creds)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code="en-US",
    )
    streaming_config = speech.StreamingRecognitionConfig(
        config=config, interim_results=True,
    )
    with MicrophoneStream() as stream:
        audio_gen = stream.generator()
        requests = (speech.StreamingRecognizeRequest(audio_content=chunk)
                    for chunk in audio_gen)
        responses = client.streaming_recognize(streaming_config, requests)
        listen_print_loop(responses)

# === Main ===
if __name__ == "__main__":
    rospy.init_node(subscriber_node_name, anonymous=True)
    mover = OdometricTurtleMover()
    rospy.Subscriber(topic_name, CompressedImage, image_callback, queue_size=1)
    threading.Thread(target=process_frames, daemon=True).start()
    threading.Thread(target=audio_thread_fn, daemon=True).start()
    rospy.loginfo("🚀 Node running: Simon says YOLO + Odometry control")
    rospy.spin()
    cv2.destroyAllWindows()
