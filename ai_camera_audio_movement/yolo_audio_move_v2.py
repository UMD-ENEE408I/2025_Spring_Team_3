#!/usr/bin/env python3

# THIS CODE DOES TURNS WELL AND DOES EVERYTHING IN v1
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
DEFAULT_DISTANCE_IN = 6  # default move distance
TURN_ANGLE_DEG = 90  # default turn angle
ANGULAR_SPEED = 0.5  # rad/s for turns
LINEAR_SPEED = 0.2  # m/s for forward/back

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
    def __init__(self, speed=LINEAR_SPEED):
        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        rospy.Subscriber("/odom", Odometry, self._odom_cb)
        self.speed = speed
        self._start_x = self._start_y = None
        self.current_x = self.current_y = None

    def _odom_cb(self, msg):
        pos = msg.pose.pose.position
        if self._start_x is None:
            self._start_x = pos.x
            self._start_y = pos.y
        self.current_x = pos.x
        self.current_y = pos.y

    def move_forward(self, inches):
        self._move_distance(inches, forward=True)

    def move_backward(self, inches):
        self._move_distance(inches, forward=False)

    def _move_distance(self, inches, forward=True):
        # wait for odometry
        rate = rospy.Rate(20)
        while self.current_x is None and not rospy.is_shutdown():
            rate.sleep()

        # compute goal
        distance_m = inches * INCH_TO_M
        start_x, start_y = self.current_x, self.current_y

        twist = Twist()
        twist.linear.x = self.speed * (1 if forward else -1)

        rospy.loginfo(
            f"🚀 Moving {'forward' if forward else 'backward'} {inches} in ({distance_m:.2f} m)"
        )
        while not rospy.is_shutdown():
            dx = self.current_x - start_x
            dy = self.current_y - start_y
            if math.hypot(dx, dy) >= distance_m:
                break
            self.cmd_pub.publish(twist)
            rate.sleep()

        # stop
        self.cmd_pub.publish(Twist())
        rospy.loginfo("🛑 Movement complete")

    def turn(self, angle_deg):
        # simple time-based turn
        duration = math.radians(angle_deg) / ANGULAR_SPEED
        twist = Twist()
        twist.angular.z = ANGULAR_SPEED if angle_deg > 0 else -ANGULAR_SPEED

        rospy.loginfo(
            f"🔄 Turning {'left' if angle_deg>0 else 'right'} {abs(angle_deg)}°"
        )
        end_time = rospy.Time.now() + rospy.Duration(duration)
        rate = rospy.Rate(10)
        while rospy.Time.now() < end_time and not rospy.is_shutdown():
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
        else:
            rospy.loginfo(f"🗣️ Heard: {transcript}")
            num_chars = 0

            # If Simon mode, look for forward/backward with optional number
            if simon_mode:
                m = re.match(r"^(?:(\d+)\s*)?(forward|backward)\b", transcript, re.I)
                if m:
                    dist = int(m.group(1)) if m.group(1) else DEFAULT_DISTANCE_IN
                    cmd = m.group(2).lower()

                    # Our movement is technically inverted so ya
                    if cmd == "forward":
                        mover.move_backward(dist)
                    else:
                        mover.move_forward(dist)
                    simon_mode = False
                    continue

                # left/right/stop as before
                if re.search(r"\bleft\b", transcript, re.I):
                    mover.turn(TURN_ANGLE_DEG)
                    simon_mode = False
                    continue
                if re.search(r"\bright\b", transcript, re.I):
                    mover.turn(-TURN_ANGLE_DEG)
                    simon_mode = False
                    continue
                if re.search(r"\bstop\b", transcript, re.I):
                    mover.move_forward(0)  # just publish zero to stop
                    simon_mode = False
                    continue

            # not in Simon mode or no match: listen for triggers
            if re.search(r"\bsimon says\b", transcript, re.I):
                rospy.loginfo("🗣️ 'Simon says' – awaiting next command")
                simon_mode = True
                continue

            if re.search(r"\b(exit|quit)\b", transcript, re.I):
                rospy.loginfo("🗣️ Exit command detected, shutting down")
                rospy.signal_shutdown("User requested exit")
                break


def audio_thread_fn():
    cfg = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code="en-US",
    )
    stream_cfg = speech.StreamingRecognitionConfig(config=cfg, interim_results=True)

    with MicrophoneStream(RATE, CHUNK) as stream:
        audio_gen = stream.generator()
        requests = (
            speech.StreamingRecognizeRequest(audio_content=chunk) for chunk in audio_gen
        )
        responses = speech_client.streaming_recognize(stream_cfg, requests)
        listen_print_loop(responses)


def main():
    rospy.init_node(subscriber_node_name, anonymous=True)
    rospy.Subscriber(topic_name, CompressedImage, image_callback, queue_size=1)

    t = threading.Thread(target=audio_thread_fn)
    t.daemon = True
    t.start()

    rospy.loginfo("🚀 Node live: camera + Simon says (with distances)")
    rospy.spin()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
