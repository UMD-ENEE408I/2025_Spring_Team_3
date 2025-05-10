#!/usr/bin/env python3
import rospy
import re
import queue
import signal
import time
import numpy as np
import cv2
import pyaudio

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CompressedImage
from tf.transformations import euler_from_quaternion
from cv_bridge import CvBridge

from google.cloud import speech
from google.oauth2 import service_account
from google.api_core.exceptions import InternalServerError

# === GLOBAL SETTINGS ===
RATE            = 16000
CHUNK           = int(RATE / 10)
TURN_ANGLE_DEG  = 90            # degrees per left/right
LINE_LOST_PAUSE = 0.05          # seconds to wait if no frame yet
DEG2RAD         = np.pi / 180.0

current_yaw    = 0.0
odom_received = False
last_frame    = None

# === ROS NODE & PUB/SUB ===
rospy.init_node('simon_says_line_follower')
cmd_pub   = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
frame_pub = rospy.Publisher('video_topic/compressed', CompressedImage, queue_size=1)

# Odometry callback
def odom_callback(msg):
    global current_yaw, odom_received
    q = msg.pose.pose.orientation
    _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
    current_yaw = yaw
    odom_received = True

rospy.Subscriber('/odom', Odometry, odom_callback)

# Camera callback (CompressedImage → OpenCV BGR)
bridge = CvBridge()
def image_cb(msg):
    global last_frame
    arr        = np.frombuffer(msg.data, np.uint8)
    last_frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

camera_topic = rospy.get_param('~camera_topic',
    '/usb_cam/image_raw/compressed')
rospy.Subscriber(camera_topic,
                 CompressedImage,
                 image_cb,
                 queue_size=1)

# === CTRL+C SHUTDOWN HANDLER ===
def shutdown_handler(signum, frame):
    rospy.loginfo("🔴 Ctrl-C pressed, shutting down")
    rospy.signal_shutdown("user interrupt")

signal.signal(signal.SIGINT, shutdown_handler)

# === UTILS ===
def normalize_angle(angle):
    """Wrap to [-π, π]."""
    return np.arctan2(np.sin(angle), np.cos(angle))

# === MICROPHONE STREAM ===
class MicrophoneStream:
    """Opens a pyaudio stream and yields raw audio chunks."""
    def __init__(self, rate=RATE, chunk=CHUNK):
        self.rate  = rate
        self.chunk = chunk
        self.buff  = queue.Queue()
        self.closed = True

    def __enter__(self):
        self.audio_interface = pyaudio.PyAudio()
        self.audio_stream    = self.audio_interface.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk,
            stream_callback=self._fill_buffer
        )
        self.closed = False
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.audio_stream.stop_stream()
        self.audio_stream.close()
        self.closed = True
        self.buff.put(None)
        self.audio_interface.terminate()

    def _fill_buffer(self, in_data, frame_count, time_info, status):
        self.buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self):
        while not self.closed:
            chunk = self.buff.get()
            if chunk is None:
                return
            data = [chunk]
            while True:
                try:
                    chunk = self.buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break
            yield b''.join(data)

# === SPEECH CLIENT SETUP ===
creds  = service_account.Credentials.from_service_account_file('key.json')
client = speech.SpeechClient(credentials=creds)

def parse_command(text):
    """Extract one of the allowed commands from text, or None."""
    for cmd in ['forward','backward','left','right','stop']:
        if re.search(rf"\b{cmd}\b", text, re.I):
            return cmd
    return None

# === MOVEMENT ===
def move_command(direction):
    """Execute one of the 5 moves: forward, backward, left, right, stop."""
    # wait for odom
    if not odom_received:
        rospy.sleep(0.1)

    twist = Twist()

    if direction == 'forward':
        twist.linear.x = -0.3
        cmd_pub.publish(twist)
        rospy.sleep(1.5)

    elif direction == 'backward':
        twist.linear.x = 0.3
        cmd_pub.publish(twist)
        rospy.sleep(1.5)

    elif direction in ('left','right'):
        # wait for valid odom
        while not odom_received and not rospy.is_shutdown():
            rospy.sleep(0.1)

        start_yaw    = current_yaw
        delta_rad    = TURN_ANGLE_DEG * DEG2RAD
        if direction=='right':
            target_yaw = normalize_angle(start_yaw - delta_rad)
            twist.angular.z = -0.3
        else:
            target_yaw = normalize_angle(start_yaw + delta_rad)
            twist.angular.z = 0.3

        rate = rospy.Rate(15)
        while not rospy.is_shutdown():
            error = normalize_angle(target_yaw - current_yaw)
            if abs(error) < 2 * DEG2RAD:  
                break
            cmd_pub.publish(twist)
            rate.sleep()

    # always send a zero to stop
    cmd_pub.publish(Twist())

# === LINE FOLLOWING ===
def follow_line():
    rospy.loginfo("▶️  Starting line-following…")
    rate = rospy.Rate(15)

    while not rospy.is_shutdown():
        if last_frame is None:
            rate.sleep()
            continue

        frame = last_frame.copy()

        # optional: republish for debugging
        resized = cv2.resize(frame, (640,480))
        msg     = bridge.cv2_to_compressed_imgmsg(resized,
                                                  dst_format='jpeg')
        frame_pub.publish(msg)

        h, w, _ = frame.shape
        roi     = frame[int(h*0.75):, :]
        gray    = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _,th    = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

        left   = np.sum(th[:, :w//3]           ==255)
        center = np.sum(th[:, w//3:2*w//3]     ==255)
        right  = np.sum(th[:, 2*w//3:]         ==255)

        # decide
        if center > max(left,right)*0.7:
            twist = Twist(); twist.linear.x = -0.03
        elif left > right:
            twist = Twist()
            twist.linear.x = 0.02
            twist.angular.z = 0.5
        elif right > left:
            twist = Twist()
            twist.linear.x = 0.02
            twist.angular.z = -0.5
        else:
            rospy.loginfo("⚠️  Line lost, awaiting Simon says")
            break

        cmd_pub.publish(twist)
        rate.sleep()

    # stop when leaving
    cmd_pub.publish(Twist())

# === VOICE LISTENER ===
def listen_for_command_once():
    config = speech.RecognitionConfig(
        encoding        = speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz = RATE,
        language_code   = 'en-US'
    )
    streaming_config = speech.StreamingRecognitionConfig(
        config         = config,
        interim_results = False
    )

    with MicrophoneStream() as stream:
        requests  = (speech.StreamingRecognizeRequest(audio_content=chunk)
                     for chunk in stream.generator())
        try:
            responses = client.streaming_recognize(streaming_config,
                                                   requests)
            for resp in responses:
                if rospy.is_shutdown():
                    return None
                if not resp.results:
                    continue
                result = resp.results[0]
                if not result.is_final:
                    continue
                txt = result.alternatives[0].transcript.lower()
                rospy.loginfo(f"Heard: {txt}")
                if 'simon says' not in txt:
                    continue
                return parse_command(txt)
        except InternalServerError:
            rospy.logwarn("⚠️  Speech API error, retrying…")
            return None

    return None

# === MAIN LOOP ===
if __name__ == '__main__':
    while not rospy.is_shutdown():
        follow_line()
        if rospy.is_shutdown():
            break

        cmd = listen_for_command_once()
        if cmd == 'stop':
            rospy.loginfo("✋ Stop received, exiting")
            break
        if cmd:
            rospy.loginfo(f"🔧 Executing command: {cmd}")
            move_command(cmd)

    rospy.loginfo("🏁 Shutting down")
    cmd_pub.publish(Twist())
