#!/usr/bin/env python3
import logging
import ctypes
from ctypes import CFUNCTYPE, c_char_p, c_int
from ctypes.util import find_library

""" 
Simon Says YOLO + Odometry ROS Node
Combines YOLO-based hand‑gesture detection and Google Cloud Speech audio commands.

Usage:
 - Say “Simon says” → next hand gesture (forward/backward/left/right) triggers
   an odometry‑based move or turn.
 - Say “audio Simon says” → next spoken command (forward/backward/left <deg>/right <deg>)
   triggers an odometry‑based move or turn.
 - Will stop during motion when "stop" is heard
 - Now we can tell it a number and it will move/turn to that much
"""

# — now import ROS, OpenCV, PyAudio, etc. —
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
DEFAULT_DISTANCE_IN = 6
TURN_ANGLE_DEG = 90
LINEAR_SPEED = 0.2
ANGULAR_SPEED = 0.5

NODE_NAME = "simon_says_yolo_odometry"
VIDEO_TOPIC = "video_topic/compressed"
YOLO_MODEL_PATH = "simonsaysv1.pt"
GOOGLE_CREDENTIALS = "key.json"

# modes & queues
simon_mode = False
audio_mode = False
frame_queue = queue.Queue(maxsize=1)
command_queue = queue.Queue()  # single-motion queue
bridge = CvBridge()

# load YOLO
model = YOLO(YOLO_MODEL_PATH)


class OdometricTurtleMover:
    def __init__(self, linear_speed=LINEAR_SPEED, angular_speed=ANGULAR_SPEED):
        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        rospy.Subscriber("/odom", Odometry, self._odom_cb)
        self.linear_speed = linear_speed
        self.angular_speed = angular_speed
        self.current_x = None
        self.current_y = None
        self.current_yaw = None
        self.stop_flag = False

    def _odom_cb(self, msg):
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        q = (ori.x, ori.y, ori.z, ori.w)
        _, _, yaw = euler_from_quaternion(q)
        self.current_x, self.current_y, self.current_yaw = pos.x, pos.y, yaw

    def stop_motion(self):
        self.stop_flag = True
        self.cmd_pub.publish(Twist())
        rospy.loginfo("⏹️ Emergency STOP: halting all motion")

    def _move_distance(self, inches, forward=True):
        rate = rospy.Rate(20)
        while self.current_x is None and not rospy.is_shutdown():
            rate.sleep()

        self.stop_flag = False
        target = inches * INCH_TO_M
        start_x, start_y = self.current_x, self.current_y
        twist = Twist()
        twist.linear.x = self.linear_speed * (1 if forward else -1)
        dir_str = "forward" if forward else "backward"
        rospy.loginfo(f'🚀 Moving {dir_str} {inches}" (~{target:.2f} m)')

        while not rospy.is_shutdown() and not self.stop_flag:
            dx = self.current_x - start_x
            dy = self.current_y - start_y
            if math.hypot(dx, dy) >= target:
                break
            self.cmd_pub.publish(twist)
            rate.sleep()

        self.cmd_pub.publish(Twist())
        rospy.loginfo("🛑 Movement complete")
        self.stop_flag = False

    def move_forward(self, inches=DEFAULT_DISTANCE_IN):
        self._move_distance(inches, forward=True)

    def move_backward(self, inches=DEFAULT_DISTANCE_IN):
        self._move_distance(inches, forward=False)

    def turn_degrees(self, angle_deg):
        rate = rospy.Rate(20)
        while self.current_yaw is None and not rospy.is_shutdown():
            rate.sleep()

        self.stop_flag = False
        start = self.current_yaw
        delta = math.radians(angle_deg)
        goal = (start + delta + math.pi) % (2 * math.pi) - math.pi
        twist = Twist()
        twist.angular.z = self.angular_speed * (1 if delta > 0 else -1)
        rospy.loginfo(f"🔄 Turning {angle_deg}° → target yaw {goal:.2f} rad")

        while not rospy.is_shutdown() and not self.stop_flag:
            err = (goal - self.current_yaw + math.pi) % (2 * math.pi) - math.pi
            if abs(err) < math.radians(2):
                break
            self.cmd_pub.publish(twist)
            rate.sleep()

        self.cmd_pub.publish(Twist())
        rospy.loginfo("🛑 Turn complete")
        self.stop_flag = False


class MicrophoneStream:
    def __init__(self, rate=RATE, chunk=CHUNK):
        self._rate, self._chunk = rate, chunk
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
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(self, in_data, frame_count, time_info, status):
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


def parse_angle(text, default=TURN_ANGLE_DEG):
    m = re.search(r"(\d+)", text)
    return int(m.group(1)) if m else default


def image_callback(msg):
    try:
        frame = bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
        if frame is not None and not frame_queue.full():
            frame_queue.put(cv2.resize(frame, (640, 480)))
    except Exception as e:
        rospy.logerr(f"❌ Frame conversion error: {e}")


def process_frames():
    global simon_mode, audio_mode
    while not rospy.is_shutdown():
        frame = frame_queue.get()
        results = model(frame, verbose=False)
        annotated = results[0].plot()
        cv2.imshow("SimonSaysYOLO", annotated)
        cv2.waitKey(1)

        if simon_mode and not audio_mode:
            for det in results[0].boxes:
                label = model.names[int(det.cls)]
                rospy.loginfo(f"🖐️ Gesture: {label}")
                if label == "forward":
                    command_queue.put((mover.move_backward, (), {}))
                elif label == "backward":
                    command_queue.put((mover.move_forward, (), {}))
                elif label == "left":
                    command_queue.put((mover.turn_degrees, (TURN_ANGLE_DEG,), {}))
                elif label == "right":
                    command_queue.put((mover.turn_degrees, (-TURN_ANGLE_DEG,), {}))
                else:
                    rospy.logwarn(f"❓ Unknown gesture: {label}")
                simon_mode = False
                break


def audio_thread():
    global simon_mode, audio_mode

    creds = service_account.Credentials.from_service_account_file(GOOGLE_CREDENTIALS)
    client = speech.SpeechClient(credentials=creds)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code="en-US",
    )
    streaming_config = speech.StreamingRecognitionConfig(
        config=config, interim_results=True
    )

    with MicrophoneStream(RATE, CHUNK) as stream:
        requests = (
            speech.StreamingRecognizeRequest(audio_content=chunk)
            for chunk in stream.generator()
        )
        responses = client.streaming_recognize(streaming_config, requests)

        for resp in responses:
            if not resp.results:
                continue
            alt = resp.results[0].alternatives
            if not alt:
                continue
            transcript = alt[0].transcript.lower().strip()
            rospy.loginfo(f"🗣️ Heard: {transcript}")

            # Stop always wins
            if "stop" in transcript:
                mover.stop_motion()
                simon_mode = False
                audio_mode = False
                continue

            # Enter audio vs gesture mode
            if "audio simon says" in transcript:
                audio_mode = True
                simon_mode = False
                rospy.loginfo("🔊 AUDIO mode on")
                continue
            if "simon says" in transcript:
                simon_mode = True
                audio_mode = False
                rospy.loginfo("🤖 GESTURE mode on")
                continue

            # Handle audio-mode commands with optional number
            if audio_mode:
                m = re.match(
                    r"^(?:(\d+)\s*)?(forward|backward|left|right)\b", transcript
                )
                if m:
                    num_str, cmd = m.groups()
                    if cmd in ("forward", "backward"):
                        dist = int(num_str) if num_str else DEFAULT_DISTANCE_IN
                        fn = (
                            mover.move_backward
                            if cmd == "forward"
                            else mover.move_forward
                        )
                        command_queue.put((fn, (dist,), {}))
                    else:
                        deg = int(num_str) if num_str else TURN_ANGLE_DEG
                        angle = deg if cmd == "left" else -deg
                        command_queue.put((mover.turn_degrees, (angle,), {}))
                    rospy.loginfo(
                        f"🛑 AUDIO mode off (did {cmd} {num_str or 'default'})"
                    )
                else:
                    rospy.logwarn(f"❓ Unknown audio cmd: {transcript}")

                simon_mode = False
                audio_mode = False


def motion_dispatcher():
    """Single-threaded worker: executes one motion at a time."""
    while not rospy.is_shutdown():
        fn, args, kwargs = command_queue.get()
        try:
            fn(*args, **kwargs)
        except Exception as e:
            rospy.logerr(f"Motion command error: {e}")
        command_queue.task_done()


def main():
    rospy.init_node(NODE_NAME, anonymous=True)
    global mover
    mover = OdometricTurtleMover()

    # start dispatcher, frame & audio threads
    threading.Thread(target=motion_dispatcher, daemon=True).start()
    threading.Thread(target=process_frames, daemon=True).start()
    threading.Thread(target=audio_thread, daemon=True).start()

    rospy.Subscriber(VIDEO_TOPIC, CompressedImage, image_callback, queue_size=1)

    rospy.loginfo("🚀 Node running: Simon Says YOLO + Odometry")
    rospy.spin()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
