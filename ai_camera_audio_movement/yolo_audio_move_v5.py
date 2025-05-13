#!/usr/bin/env python3
"""
Simon Says YOLO + Odometry ROS Node
Combines YOLO-based hand‑gesture detection and Google Cloud Speech audio commands.

Usage:
 - Say “Simon says” → next hand gesture (forward/backward/left/right) triggers
   an odometry‑based move or turn.
 - Say “audio Simon says” → next spoken command (forward/backward/left <deg>/right <deg>)
   triggers an odometry‑based move or turn.
"""

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
RATE = 16000  # Audio sample rate
CHUNK = int(RATE / 10)  # 100ms
INCH_TO_M = 0.0254  # Inches to meters
DEFAULT_DISTANCE_IN = 6  # Default move distance in inches
TURN_ANGLE_DEG = 90  # Default turn angle in degrees
LINEAR_SPEED = 0.2  # m/s
ANGULAR_SPEED = 0.5  # rad/s

subscriber_node_name = "simon_says_yolo_odometry"
topic_name = "video_topic/compressed"

# === Global state ===
simon_mode = False  # True after “Simon says” (vision)
audio_mode = False  # True after “audio Simon says” (voice)
frame_queue = queue.Queue(maxsize=1)
bridge = CvBridge()

# Load your YOLO model (adjust path as needed)
model = YOLO("simonsaysv1.pt")


class OdometricTurtleMover:
    """Uses /odom feedback to move a differential‑drive robot precisely."""

    def __init__(self, linear_speed=LINEAR_SPEED, angular_speed=ANGULAR_SPEED):
        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        rospy.Subscriber("/odom", Odometry, self._odom_cb)
        self.linear_speed = linear_speed
        self.angular_speed = angular_speed

        # odometry state
        self.current_x = None
        self.current_y = None
        self.current_yaw = None

    def _odom_cb(self, msg):
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        quat = (ori.x, ori.y, ori.z, ori.w)
        _, _, yaw = euler_from_quaternion(quat)
        self.current_x = pos.x
        self.current_y = pos.y
        self.current_yaw = yaw

    def _move_distance(self, inches, forward=True):
        """Move forward/backward a given number of inches."""
        rate = rospy.Rate(20)
        # wait for valid odom
        while self.current_x is None and not rospy.is_shutdown():
            rate.sleep()

        distance_m = inches * INCH_TO_M
        start_x, start_y = self.current_x, self.current_y
        twist = Twist()
        twist.linear.x = self.linear_speed * (1 if forward else -1)
        dir_str = "forward" if forward else "backward"
        rospy.loginfo(f"🚀 Moving {dir_str} {inches} in (~{distance_m:.2f} m)")

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
        self._move_distance(inches, forward=True)

    def move_backward(self, inches=DEFAULT_DISTANCE_IN):
        self._move_distance(inches, forward=False)

    def turn_degrees(self, angle_deg):
        """Turn in place by a given angle in degrees (+ left, – right)."""
        rate = rospy.Rate(20)
        # wait for yaw
        while self.current_yaw is None and not rospy.is_shutdown():
            rate.sleep()

        start = self.current_yaw
        delta = math.radians(angle_deg)
        # normalize goal to [−π, π]
        goal = (start + delta + math.pi) % (2 * math.pi) - math.pi

        twist = Twist()
        twist.angular.z = self.angular_speed * (1 if delta > 0 else -1)
        rospy.loginfo(f"🔄 Turning {angle_deg}° (to yaw {goal:.2f} rad)")
        while not rospy.is_shutdown():
            err = (goal - self.current_yaw + math.pi) % (2 * math.pi) - math.pi
            if abs(err) < math.radians(2):
                break
            self.cmd_pub.publish(twist)
            rate.sleep()

        self.cmd_pub.publish(Twist())
        rospy.loginfo("🛑 Turn complete")


class MicrophoneStream:
    """Opens a blocking-streaming interface to the microphone."""

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

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        # signal termination
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        """Continuously collect audio chunks into buffer."""
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self):
        """Yield PCM audio chunks as they become available."""
        while not self.closed:
            chunk = self._buff.get()
            if chunk is None:
                return
            data = [chunk]
            # grab any extra data in buffer
            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break
            yield b"".join(data)


def parse_angle(text, default=TURN_ANGLE_DEG):
    """Extract the first integer found in text, or return default."""
    m = re.search(r"(\d+)", text)
    return int(m.group(1)) if m else default


def image_callback(msg):
    """ROS subscriber callback: enqueue each frame for YOLO processing."""
    try:
        frame = bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
        if frame is not None:
            resized = cv2.resize(frame, (640, 480))
            if not frame_queue.full():
                frame_queue.put(resized)
    except Exception as e:
        rospy.logerr(f"❌ Error converting frame: {e}")


def process_frames():
    """Continuously run YOLO on new frames; if in gesture‑mode, dispatch moves."""
    global simon_mode, audio_mode
    while not rospy.is_shutdown():
        frame = frame_queue.get()
        results = model(frame)
        annotated = results[0].plot()
        cv2.imshow("SimonSaysYOLO", annotated)
        cv2.waitKey(1)

        if simon_mode and not audio_mode:
            for det in results[0].boxes:
                label = model.names[int(det.cls)]
                rospy.loginfo(f"🖐️ Gesture detected: {label}")
                if label == "forward":
                    mover.move_backward()
                elif label == "backward":
                    mover.move_forward()
                elif label == "left":
                    mover.turn_degrees(TURN_ANGLE_DEG)
                elif label == "right":
                    mover.turn_degrees(-TURN_ANGLE_DEG)
                else:
                    rospy.logwarn(f"❓ Unknown gesture: {label}")
                # consume the gesture
                simon_mode = False
                break


def audio_thread():
    """Continuously listen to microphone; switch modes and dispatch voice commands."""
    global simon_mode, audio_mode

    creds = service_account.Credentials.from_service_account_file("key.json")
    speech_client = speech.SpeechClient(credentials=creds)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code="en-US",
    )
    streaming_config = speech.StreamingRecognitionConfig(
        config=config, interim_results=True
    )

    with MicrophoneStream(RATE, CHUNK) as stream:
        audio_generator = stream.generator()
        requests = (
            speech.StreamingRecognizeRequest(audio_content=chunk)
            for chunk in audio_generator
        )
        responses = speech_client.streaming_recognize(streaming_config, requests)

        for resp in responses:
            if not resp.results:
                continue
            result = resp.results[0]
            if not result.alternatives:
                continue
            transcript = result.alternatives[0].transcript.lower()
            rospy.loginfo(f"🗣️ Heard: {transcript}")

            # Mode switching
            if "audio simon says" in transcript:
                audio_mode = True
                simon_mode = False
                rospy.loginfo("🔊 Entered AUDIO mode — next speech will be command")
                continue
            if "simon says" in transcript:
                simon_mode = True
                audio_mode = False
                rospy.loginfo("🤖 Entered GESTURE mode — next hand sign will move")
                continue

            # If in audio_mode, interpret command
            if audio_mode:
                if "forward" in transcript:
                    mover.move_forward()
                elif "backward" in transcript:
                    mover.move_backward()
                elif "left" in transcript:
                    ang = parse_angle(transcript)
                    mover.turn_degrees(ang)
                elif "right" in transcript:
                    ang = parse_angle(transcript)
                    mover.turn_degrees(-ang)
                else:
                    rospy.logwarn(f"❓ Unrecognized audio cmd: {transcript}")
                # reset modes
                simon_mode = False
                audio_mode = False
                rospy.loginfo("🛑 Exiting AUDIO mode")


def main():
    rospy.init_node(subscriber_node_name, anonymous=True)
    # Instantiate mover (subscribes to /odom internally)
    global mover
    mover = OdometricTurtleMover()

    # Subscribe to camera frames
    rospy.Subscriber(topic_name, CompressedImage, image_callback, queue_size=1)

    # Start processing threads
    threading.Thread(target=process_frames, daemon=True).start()
    threading.Thread(target=audio_thread, daemon=True).start()

    rospy.loginfo("🚀 Node live: Simon Says YOLO + Odometry")
    rospy.spin()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
