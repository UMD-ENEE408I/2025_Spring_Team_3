#!/usr/bin/env python3
import logging
import ctypes
from ctypes.util import find_library

# ——— Graceful ALSA warning suppression ———
libname = find_library('asound')
if libname:
    try:
        asound = ctypes.cdll.LoadLibrary(libname)
        # Prototype: void handler(const char*, int, const char*, int, const char*)
        ERRFUNC = ctypes.CFUNCTYPE(
            None,
            ctypes.c_char_p,  # filename
            ctypes.c_int,     # line
            ctypes.c_char_p,  # function
            ctypes.c_int,     # err
            ctypes.c_char_p   # fmt
        )
        def _alsa_error_handler(fname, line, func, err, fmt):
            # no‑op
            return
        c_handler = ERRFUNC(_alsa_error_handler)
        asound.snd_lib_error_set_handler(c_handler)
        logging.info(f"✔ ALSA errors suppressed via {libname}")
    except (AttributeError, OSError) as e:
        logging.warning(f"Could not install ALSA error handler: {e}")
else:
    logging.warning("libasound not found; ALSA warnings will still print")

# — now import ROS, OpenCV, PyAudio, etc. —
import rospy
import cv2
import re
import threading
import queue
import math
from ultralytics import YOLO
from sensor_msgs.msg import CompressedImage, Odometry
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
from tf.transformations import euler_from_quaternion
from google.cloud import speech
from google.oauth2 import service_account
import pyaudio

# === Suppress only ALSA errors (not all stderr) ===
# Define the C prototype for the error handler
ERROR_HANDLER_FUNC = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)


def _alsa_error_handler(filename, line, function, err, fmt):
    # no-op
    pass


# Install the handler
asound = ctypes.cdll.LoadLibrary("libasound.so")
asound.snd_lib_error_set_handler(ERROR_HANDLER_FUNC(_alsa_error_handler))


# === Constants ===
RATE = 16000  # Audio sample rate
CHUNK = int(RATE / 10)  # 100ms
INCH_TO_M = 0.0254  # Inches to meters
DEFAULT_DISTANCE_IN = 6  # Default move distance in inches
TURN_ANGLE_DEG = 90  # Default turn angle in degrees
LINEAR_SPEED = 0.2  # m/s
ANGULAR_SPEED = 0.5  # rad/s

NODE_NAME = "simon_says_yolo_odometry"
VIDEO_TOPIC = "video_topic/compressed"
YOLO_MODEL_PATH = "simonsaysv1.pt"
GOOGLE_CREDENTIALS = "key.json"

# modes & queues
simon_mode = False  # for vision-based moves
audio_mode = False  # for voice-based moves
frame_queue = queue.Queue(maxsize=1)
bridge = CvBridge()

# load YOLO
model = YOLO(YOLO_MODEL_PATH)


class OdometricTurtleMover:
    """Uses /odom feedback to move, with an emergency stop override."""

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
        self.current_x = pos.x
        self.current_y = pos.y
        self.current_yaw = yaw

    def stop_motion(self):
        """Emergency stop: abort any ongoing move/turn immediately."""
        self.stop_flag = True
        self.cmd_pub.publish(Twist())
        rospy.loginfo("⏹️ Emergency STOP: halting all motion")

    def _move_distance(self, inches, forward=True):
        rate = rospy.Rate(20)
        # wait until we have odom data
        while self.current_x is None and not rospy.is_shutdown():
            rate.sleep()

        self.stop_flag = False
        distance_m = inches * INCH_TO_M
        start_x, start_y = self.current_x, self.current_y
        twist = Twist()
        twist.linear.x = self.linear_speed * (1 if forward else -1)
        dir_str = "forward" if forward else "backward"
        rospy.loginfo(f'🚀 Moving {dir_str} {inches}" (~{distance_m:.2f} m)')

        while not rospy.is_shutdown() and not self.stop_flag:
            dx = self.current_x - start_x
            dy = self.current_y - start_y
            if math.hypot(dx, dy) >= distance_m:
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
        start_yaw = self.current_yaw
        delta = math.radians(angle_deg)
        goal = (start_yaw + delta + math.pi) % (2 * math.pi) - math.pi
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
    """Yield raw PCM chunks from the microphone."""

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
    """Enqueue each camera frame for YOLO processing."""
    try:
        frame = bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
        if frame is not None:
            resized = cv2.resize(frame, (640, 480))
            if not frame_queue.full():
                frame_queue.put(resized)
    except Exception as e:
        rospy.logerr(f"❌ Frame conversion error: {e}")


def process_frames():
    """Continuously run YOLO; in simon_mode, dispatch moves on gestures."""
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
                rospy.loginfo(f"🖐️ Gesture: {label}")
                if label == "forward":
                    mover.move_forward()
                elif label == "backward":
                    mover.move_backward()
                elif label == "left":
                    mover.turn_degrees(TURN_ANGLE_DEG)
                elif label == "right":
                    mover.turn_degrees(-TURN_ANGLE_DEG)
                else:
                    rospy.logwarn(f"❓ Unknown gesture: {label}")
                simon_mode = False
                break


def audio_thread():
    """Continuously listen, toggle modes, and execute audio commands."""
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

            # Emergency stop ➔ always wins
            if "stop" in transcript:
                mover.stop_motion()
                simon_mode = False
                audio_mode = False
                continue

            # Mode switches
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

            # If in audio_mode, dispatch the spoken command
            if audio_mode:
                if "forward" in transcript:
                    mover.move_forward()
                elif "backward" in transcript:
                    mover.move_backward()
                elif "left" in transcript:
                    mover.turn_degrees(parse_angle(transcript))
                elif "right" in transcript:
                    mover.turn_degrees(-parse_angle(transcript))
                else:
                    rospy.logwarn(f"❓ Unknown audio cmd: {transcript}")
                simon_mode = False
                audio_mode = False
                rospy.loginfo("🛑 AUDIO mode off")


def main():
    rospy.init_node(NODE_NAME, anonymous=True)
    global mover
    mover = OdometricTurtleMover()

    rospy.Subscriber(VIDEO_TOPIC, CompressedImage, image_callback, queue_size=1)

    threading.Thread(target=process_frames, daemon=True).start()
    threading.Thread(target=audio_thread, daemon=True).start()

    rospy.loginfo("🚀 Node running: Simon Says YOLO + Odometry")
    rospy.spin()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
