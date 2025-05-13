#!/usr/bin/env python3

import rospy
import cv2
import queue
import time
import pyaudio
import torch
from threading import Thread
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
from ultralytics import YOLO
from google.cloud import speech
from google.oauth2 import service_account

# === Constants ===
RATE = 16000
CHUNK = int(RATE / 10)
GESTURES = ['forward', 'backward', 'left', 'right', 'stop']

# === Globals ===
pub = None
bridge = CvBridge()
frame_queue = queue.Queue(maxsize=2)
latest_frame = None
frame_ready = False
frame_count = 0

# === Camera topic and model path ===
camera_topic = "/video_topic"  # Uncompressed image, based on your working pub
model_path = "simonsaysv1.pt"

# === Movement Command ===
def move_command(direction):
    twist = Twist()
    duration = 1.5

    if direction == "forward":
        twist.linear.x = 0.3
    elif direction == "backward":
        twist.linear.x = -0.3
    elif direction == "left":
        twist.angular.z = 1.5
        duration = 1.0
    elif direction == "right":
        twist.angular.z = -1.5
        duration = 1.0
    elif direction == "stop":
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        pub.publish(twist)
        return

    rospy.loginfo(f"🦾 Executing gesture: {direction}")
    pub.publish(twist)
    time.sleep(duration)
    pub.publish(Twist())

# === Microphone Stream Class ===
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
        return self

    def __exit__(self, type, value, traceback):
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

# === Frame Callback ===
def camera_callback(msg):
    global latest_frame, frame_count, frame_ready
    try:
        frame = bridge.imgmsg_to_cv2(msg, "bgr8")
        frame = cv2.resize(frame, (640, 480))
        latest_frame = frame

        cv2.imshow("📷 Live Feed", frame)
        if cv2.waitKey(1) == 27:
            rospy.signal_shutdown("ESC pressed")

        frame_count += 1
        if frame_count > 10:
            frame_ready = True

        if not frame_queue.full():
            frame_queue.put(frame)
    except Exception as e:
        rospy.logerr(f"❌ Frame callback error: {e}")

# === YOLO Gesture Detection ===
def detect_gesture(model):
    timeout = time.time() + 5
    while time.time() < timeout:
        if not frame_queue.empty():
            frame = frame_queue.get()
            results = model(frame)
            boxes = results[0].boxes
            for box in boxes:
                if box.conf[0] > 0.5:
                    label = results[0].names[int(box.cls[0])]
                    if label in GESTURES:
                        rospy.loginfo(f"🧠 Detected gesture: {label}")
                        return label
    rospy.logwarn("⚠️ No gesture detected in time.")
    return None

# === Main loop for speech → vision → motion
def audio_yolo_loop():
    creds = service_account.Credentials.from_service_account_file("key.json")
    client = speech.SpeechClient(credentials=creds)

    model = YOLO(model_path)

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code="en-US",
    )
    streaming_config = speech.StreamingRecognitionConfig(
        config=config,
        interim_results=True,
    )

    with MicrophoneStream(RATE, CHUNK) as stream:
        audio_generator = stream.generator()
        requests = (speech.StreamingRecognizeRequest(audio_content=content) for content in audio_generator)
        responses = client.streaming_recognize(streaming_config, requests)

        rospy.loginfo("🎤 Listening for 'Simon says'...")

        simon_triggered = False
        for response in responses:
            if not response.results or not response.results[0].alternatives:
                continue
            transcript = response.results[0].alternatives[0].transcript.lower()
            rospy.loginfo(f"📝 Heard: {transcript}")

            if not simon_triggered and "simon says" in transcript:
                rospy.loginfo("🗣️ Simon says detected! Waiting for gesture...")
                simon_triggered = True
                gesture = detect_gesture(model)
                if gesture:
                    move_command(gesture)
                simon_triggered = False

# === Entry point
if __name__ == "__main__":
    rospy.init_node("simon_says_combined_node")
    pub = rospy.Publisher("cmd_vel", Twist, queue_size=10)
    rospy.Subscriber(camera_topic, Image, camera_callback, queue_size=1)

    Thread(target=rospy.spin, daemon=True).start()

    # Wait for camera frames before launching audio + YOLO
    rospy.loginfo("⏳ Waiting for camera stream...")
    start = time.time()
    while not frame_ready and time.time() - start < 10:
        time.sleep(0.1)

    if not frame_ready:
        rospy.logerr("❌ No camera feed detected. Exiting.")
        exit()

    rospy.loginfo("✅ Camera feed ready. Launching audio and vision loop.")
    try:
        audio_yolo_loop()
    except KeyboardInterrupt:
        rospy.loginfo("🔻 Shutting down.")
    finally:
        cv2.destroyAllWindows()
        pub.publish(Twist())
