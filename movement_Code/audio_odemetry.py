#!/usr/bin/env python3

import os
import re
import sys
import time
import queue
import math
import signal
import rospy

import pyaudio
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
from google.cloud import speech
from google.oauth2 import service_account

# === CTRL+C SHUTDOWN HANDLER ===
def shutdown_handler(signum, frame):
    print("🔴 Shutdown signal received. Stopping robot and exiting.")
    rospy.signal_shutdown("KeyboardInterrupt")

signal.signal(signal.SIGINT, shutdown_handler)

# === GLOBAL SETTINGS ===
TURN_ANGLE_DEG = 90
RATE = 16000
CHUNK = int(RATE / 10)
current_yaw = 0.0

# === GOOGLE CLOUD SETUP ===
creds = service_account.Credentials.from_service_account_file("key.json")
client = speech.SpeechClient(credentials=creds)

# === AUDIO STREAM SETUP ===
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

# === ODOMETRY ===
def odom_callback(msg):
    global current_yaw
    orientation_q = msg.pose.pose.orientation
    _, _, yaw = euler_from_quaternion(
        [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
    )
    current_yaw = yaw

def normalize_angle(angle):
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle

# === ROBOT MOTION ===
def move_command(pub, direction):
    global current_yaw
    twist = Twist()
    rospy.sleep(0.1)

    if direction in ["left", "right"]:
        if direction == "right":
            target_angle_rad = math.radians(TURN_ANGLE_DEG * 0.95)
            target_angle_rad *= -1
        else:
            target_angle_rad = math.radians(TURN_ANGLE_DEG * 1.06)

        reset_offset = current_yaw
        def get_relative_yaw():
            return normalize_angle(current_yaw - reset_offset)

        start_yaw = 0.0
        end_yaw = normalize_angle(start_yaw + target_angle_rad)

        print(f"\n➡️ Turning {direction.upper()} by ~{abs(math.degrees(target_angle_rad)):.2f}° (relative)")
        print(f"🔁 Reset yaw: {math.degrees(reset_offset):.2f}°")
        print(f"🎯 Target relative yaw: {math.degrees(end_yaw):.2f}°\n")

        rate = rospy.Rate(10)
        start_time = time.time()

        while not rospy.is_shutdown():
            elapsed = time.time() - start_time
            relative_yaw = get_relative_yaw()
            angle_diff = normalize_angle(end_yaw - relative_yaw)

            print(
                f"🧭 Current yaw: {math.degrees(current_yaw):.2f}°, "
                f"Relative: {math.degrees(relative_yaw):.2f}°, "
                f"Remaining: {math.degrees(angle_diff):.2f}°"
            )

            if abs(angle_diff) < 0.15:
                pub.publish(Twist())
                print("✅ Target reached.\n")
                break

            if elapsed > 10:
                pub.publish(Twist())
                print("⏰ Timeout — stopping early.\n")
                break

            if abs(angle_diff) < 0.5:
                speed = 0.035
            else:
                speed = max(0.08, min(0.4, abs(angle_diff)))
            twist.angular.z = speed if angle_diff > 0 else -speed

            pub.publish(twist)
            rate.sleep()

    elif direction == "forward":
        twist.linear.x = -0.3
        pub.publish(twist)
        time.sleep(1.5)
    elif direction == "backward":
        twist.linear.x = 0.3
        pub.publish(twist)
        time.sleep(1.5)

    pub.publish(Twist())

# === SPEECH HANDLING ===
def parse_command(transcript):
    for cmd in ["forward", "backward", "left", "right"]:
        if re.search(rf"\b{cmd}\b", transcript, re.I):
            return cmd
    return None

def listen_and_execute(pub):
    language_code = "en-US"
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code=language_code,
    )
    streaming_config = speech.StreamingRecognitionConfig(
        config=config, interim_results=False
    )

    with MicrophoneStream(RATE, CHUNK) as stream:
        audio_generator = stream.generator()
        requests = (
            speech.StreamingRecognizeRequest(audio_content=content)
            for content in audio_generator
        )
        responses = client.streaming_recognize(streaming_config, requests)

        print("🎤 Listening for 'Simon says'...")

        simon_triggered = False
        try:
            for response in responses:
                if rospy.is_shutdown():
                    break
                if not response.results:
                    continue

                result = response.results[0]
                if not result.is_final:
                    continue

                transcript = result.alternatives[0].transcript.lower()
                print("📝 Heard:", transcript)

                if not simon_triggered and "simon says" in transcript:
                    print("🗣️ Simon says detected! Waiting for command...")
                    simon_triggered = True
                    continue

                if simon_triggered:
                    command = parse_command(transcript)
                    if command:
                        move_command(pub, command)
                        simon_triggered = False
                        print("✅ Ready for next 'Simon says'")
                    elif "exit" in transcript or "quit" in transcript:
                        print("👋 Exiting...")
                        break
        except KeyboardInterrupt:
            print("🛑 Ctrl+C detected in speech loop. Exiting...")
        finally:
            stream.closed = True

# === MAIN ===
if __name__ == "__main__":
    print("🚀 Starting Simon Says Voice Control")
    rospy.init_node("simon_says_teleop")
    pub = rospy.Publisher("cmd_vel", Twist, queue_size=10)
    rospy.Subscriber("/odom", Odometry, odom_callback)

    try:
        listen_and_execute(pub)
    except KeyboardInterrupt:
        print("🛑 Interrupted by user")
    finally:
        pub.publish(Twist())
