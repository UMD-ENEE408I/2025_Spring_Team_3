#!/usr/bin/env python3

# THIS CODE DOES  DOES TURNS ACCORDING TO SPEECH INPUT 
# AND EVERYTHIGN IN v2
import rospy
import cv2
import threading
import queue
import re
import sys
import math
from sensor_msgs.msg import CompressedImage
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
from tf.transformations import euler_from_quaternion
from google.cloud import speech
from google.oauth2 import service_account
import pyaudio

# === Constants ===
INCH_TO_M = 0.0254
DEFAULT_DISTANCE_IN = 6    # default move distance in inches
TURN_ANGLE_DEG = 90        # default turn angle in degrees
ANGULAR_SPEED = 0.5        # rad/s for turns
LINEAR_SPEED = 0.2         # m/s for forward/backward

# === ROS / Camera Setup ===
subscriber_node_name = "camera_sensor_subscriber"
topic_name = "video_topic/compressed"
bridge = CvBridge()
_first_frame_logged = False

def image_callback(msg):
    global _first_frame_logged
    if not _first_frame_logged:
        rospy.loginfo("📷 Received first video frame")
        _first_frame_logged = True
    try:
        frame = bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
        cv2.imshow("camera", frame)
        cv2.waitKey(1)
    except Exception as e:
        rospy.logerr(f"Error displaying frame: {e}")

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

    def _move_distance(self, inches, forward=True):
        # wait for odometry
        rate = rospy.Rate(20)
        while self.current_x is None and not rospy.is_shutdown():
            rate.sleep()

        distance_m = inches * INCH_TO_M
        start_x, start_y = self.current_x, self.current_y
        twist = Twist()
        twist.linear.x = self.linear_speed * (1 if forward else -1)

        dir_str = 'forward' if forward else 'backward'
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

    def move_forward(self, inches):
        self._move_distance(inches, forward=True)

    def move_backward(self, inches):
        self._move_distance(inches, forward=False)

    def turn_degrees(self, angle_deg):
        # wait for yaw
        rate = rospy.Rate(20)
        while self.current_yaw is None and not rospy.is_shutdown():
            rate.sleep()

        start = self.current_yaw
        delta = math.radians(angle_deg)
        goal = (start + delta + math.pi) % (2*math.pi) - math.pi

        twist = Twist()
        twist.angular.z = self.angular_speed * (1 if delta > 0 else -1)
        rospy.loginfo(f"🔄 Turning {angle_deg}° (to yaw {goal:.2f} rad)")
        while not rospy.is_shutdown():
            err = (goal - self.current_yaw + math.pi) % (2*math.pi) - math.pi
            if abs(err) < math.radians(2):
                break
            self.cmd_pub.publish(twist)
            rate.sleep()

        self.cmd_pub.publish(Twist())
        rospy.loginfo("🛑 Turn complete")

# instantiate mover
mover = OdometricTurtleMover()

# === Audio Setup ===
RATE = 16000
CHUNK = int(RATE / 10)
creds = service_account.Credentials.from_service_account_file("key.json")
speech_client = speech.SpeechClient(credentials=creds)
simon_mode = False

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
    num_chars = 0

    for response in responses:
        if not response.results:
            continue
        result = response.results[0]
        if not result.alternatives:
            continue

        transcript = result.alternatives[0].transcript.strip()
        overwrite = " " * max(0, num_chars - len(transcript))

        if not result.is_final:
            sys.stdout.write(transcript + overwrite + "\r")
            sys.stdout.flush()
            num_chars = len(transcript)
            continue

        rospy.loginfo(f"🗣️ Heard: {transcript}")
        num_chars = 0

        # 1) Single-utterance: “Simon says <opt number> <command>”
        m_comb = re.match(
            r'(?i)\bsimon says\b\s*(?:(\d+)\s*)?'
            r'(forward|backward|left|right|stop)\b',
            transcript
        )
        if m_comb:
            num_str, cmd = m_comb.groups()
            cmd = cmd.lower()
            if cmd in ('forward','backward'):
                dist = int(num_str) if num_str else DEFAULT_DISTANCE_IN
            else:
                deg  = int(num_str) if num_str else TURN_ANGLE_DEG

            if cmd == 'forward':
                mover.move_backward(dist)
            elif cmd == 'backward':
                mover.move_forward(dist)
            elif cmd == 'left':
                mover.turn_degrees(deg)
            elif cmd == 'right':
                mover.turn_degrees(-deg)
            else:
                mover.move_forward(0)
            simon_mode = False
            continue

        # 2) Two-step: if awaiting command
        if simon_mode:
            m_two = re.match(r'^(?:(\d+)\s*)?'
                             r'(forward|backward|left|right|stop)\b',
                             transcript, re.I)
            if m_two:
                num_str, cmd = m_two.groups()
                cmd = cmd.lower()
                if cmd in ('forward','backward'):
                    dist = int(num_str) if num_str else DEFAULT_DISTANCE_IN
                else:
                    deg  = int(num_str) if num_str else TURN_ANGLE_DEG

                if cmd == 'forward':
                    mover.move_backward(dist)
                elif cmd == 'backward':
                    mover.move_forward(dist)
                elif cmd == 'left':
                    mover.turn_degrees(deg)
                elif cmd == 'right':
                    mover.turn_degrees(-deg)
                else:
                    mover.move_forward(0)
                simon_mode = False
                continue

        # 3) Trigger “Simon says” by itself
        if re.search(r'(?i)\bsimon says\b', transcript):
            rospy.loginfo("🗣️ 'Simon says' – awaiting next command")
            simon_mode = True
            continue

        # 4) Exit/Quit
        if re.search(r'(?i)\b(exit|quit)\b', transcript):
            rospy.loginfo("🗣️ Exit command detected, shutting down")
            rospy.signal_shutdown("User requested exit")
            break

def audio_thread_fn():
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code="en-US",
    )
    stream_cfg = speech.StreamingRecognitionConfig(
        config=config, interim_results=True
    )

    with MicrophoneStream(RATE, CHUNK) as stream:
        audio_gen = stream.generator()
        requests = (
            speech.StreamingRecognizeRequest(audio_content=chunk)
            for chunk in audio_gen
        )
        responses = speech_client.streaming_recognize(stream_cfg, requests)
        listen_print_loop(responses)

def main():
    rospy.init_node(subscriber_node_name, anonymous=True)
    rospy.Subscriber(topic_name, CompressedImage, image_callback, queue_size=1)

    t = threading.Thread(target=audio_thread_fn)
    t.daemon = True
    t.start()

    rospy.loginfo("🚀 Node live: camera + Simon says (with distances & degrees)")
    rospy.spin()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
