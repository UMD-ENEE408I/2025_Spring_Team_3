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
RATE           = 16000
CHUNK          = int(RATE/10)
TURN_ANGLE_DEG = 90
DEG2RAD        = np.pi/180.0

current_yaw    = 0.0
odom_received  = False
last_frame     = None

# === ROS NODE & PUB/SUB ===
rospy.init_node('simon_says_line_follower')
cmd_pub   = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
frame_pub = rospy.Publisher('video_topic/compressed', CompressedImage, queue_size=1)

# Odometry callback
def odom_callback(msg):
    global current_yaw, odom_received
    q = msg.pose.pose.orientation
    _, _, current_yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
    odom_received = True

rospy.Subscriber('/odom', Odometry, odom_callback)

# Camera callback (CompressedImage → OpenCV BGR)
bridge = CvBridge()
def image_cb(msg):
    global last_frame
    arr        = np.frombuffer(msg.data, np.uint8)
    last_frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

camera_topic = rospy.get_param('~camera_topic', '/camera/image/compressed')
rospy.Subscriber(camera_topic, CompressedImage, image_cb, queue_size=1)

# === CTRL-C HANDLER ===
def shutdown_handler(signum, frame):
    rospy.loginfo("🔴 Ctrl‑C pressed, shutting down")
    rospy.signal_shutdown("user interrupt")

signal.signal(signal.SIGINT, shutdown_handler)

# === UTILS ===
def normalize_angle(a):
    return np.arctan2(np.sin(a), np.cos(a))

# === MICROPHONE STREAM ===
class MicrophoneStream:
    def __init__(self, rate=RATE, chunk=CHUNK):
        self.rate, self.chunk = rate, chunk
        self.buff = queue.Queue()
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

    def __exit__(self, *args):
        self.audio_stream.stop_stream()
        self.audio_stream.close()
        self.closed = True
        self.buff.put(None)
        self.audio_interface.terminate()

    def _fill_buffer(self, in_data, frame_count, t1, status):
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

# === SPEECH CLIENT ===
creds  = service_account.Credentials.from_service_account_file('key.json')
client = speech.SpeechClient(credentials=creds)

def parse_cmd(txt):
    for c in ('forward','backward','left','right','stop'):
        if re.search(rf"\b{c}\b", txt, re.I):
            return c
    return None

# === MOVEMENT ===
def move_command(direction):
    if not odom_received:
        rospy.sleep(0.1)
    twist = Twist()
    if direction=='forward':
        twist.linear.x = -0.3; cmd_pub.publish(twist); rospy.sleep(1.5)
    elif direction=='backward':
        twist.linear.x = 0.3;  cmd_pub.publish(twist); rospy.sleep(1.5)
    else:
        # left/right
        while not odom_received and not rospy.is_shutdown():
            rospy.sleep(0.1)
        start = current_yaw
        delta = TURN_ANGLE_DEG*DEG2RAD
        if direction=='left':
            target = normalize_angle(start+delta);  twist.angular.z = 0.3
        else:
            target = normalize_angle(start-delta);  twist.angular.z = -0.3
        rate = rospy.Rate(15)
        while not rospy.is_shutdown():
            err = normalize_angle(target-current_yaw)
            if abs(err)<2*DEG2RAD:
                break
            cmd_pub.publish(twist)
            rate.sleep()
    cmd_pub.publish(Twist())

# === LINE FOLLOWING ===
def follow_line():
    rospy.loginfo("▶️ Starting line‑following")
    rate = rospy.Rate(15)
    while not rospy.is_shutdown():
        if last_frame is None:
            rate.sleep()
            continue
        frame = last_frame.copy()
        # republish debug
        msg   = bridge.cv2_to_compressed_imgmsg(cv2.resize(frame,(640,480)),
                                                dst_format='jpeg')
        frame_pub.publish(msg)
        h,w,_ = frame.shape
        roi   = frame[int(h*0.75):]
        gray  = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
        _,th  = cv2.threshold(gray,150,255,cv2.THRESH_BINARY)
        left   = np.sum(th[:,:w//3]==255)
        centre = np.sum(th[:,w//3:2*w//3]==255)
        right  = np.sum(th[:,2*w//3:]==255)
        if centre>max(left,right)*0.7:
            t=Twist(); t.linear.x=-0.03
        elif left>right:
            t=Twist(); t.linear.x=0.02; t.angular.z=0.5
        elif right>left:
            t=Twist(); t.linear.x=0.02; t.angular.z=-0.5
        else:
            rospy.loginfo("⚠️ Line lost, awaiting Simon says…")
            break
        cmd_pub.publish(t)
        rate.sleep()
    cmd_pub.publish(Twist())

# === TWO‑PHASE VOICE LISTENER ===
def listen_for_simon_command():
    config = speech.RecognitionConfig(
        encoding         = speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz= RATE,
        language_code    = 'en-US'
    )
    streaming = speech.StreamingRecognitionConfig(
        config         = config,
        interim_results= False
    )

    rospy.loginfo("🔈 now listening for 'Simon says'")
    with MicrophoneStream() as mic:
        requests = (speech.StreamingRecognizeRequest(audio_content=c)
                    for c in mic.generator())
        try:
            responses = client.streaming_recognize(streaming, requests)
            heard_simon = False
            for r in responses:
                if rospy.is_shutdown(): return None
                if not r.results or not r.results[0].alternatives:
                    continue
                txt = r.results[0].alternatives[0].transcript.lower()
                rospy.loginfo(f"👂 {txt}")
                if not heard_simon:
                    if 'simon says' in txt:
                        rospy.loginfo("🗣️ Simon says detected; now waiting for your command")
                        heard_simon = True
                    continue
                # second phase
                cmd = parse_cmd(txt)
                if cmd:
                    return cmd
        except InternalServerError:
            rospy.logwarn("⚠️ Speech API error—retrying next cycle")
            return None
    return None

# === MAIN LOOP ===
if __name__=='__main__':
    while not rospy.is_shutdown():
        follow_line()
        if rospy.is_shutdown(): break

        cmd = listen_for_simon_command()
        if cmd=='stop':
            rospy.loginfo("✋ Stop received, exiting")
            break
        if cmd:
            rospy.loginfo(f"🔧 Executing command: {cmd}")
            move_command(cmd)

    rospy.loginfo("🏁 Shutting down")
    cmd_pub.publish(Twist())
