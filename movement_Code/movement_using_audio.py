#!/usr/bin/env python3

import os
import re
import sys
import time
import queue
import select
import termios
import tty
import rospy

import pyaudio
from geometry_msgs.msg import Twist
from google.cloud import speech
from google.oauth2 import service_account

# === GOOGLE CLOUD SETUP ===
creds = service_account.Credentials.from_service_account_file(
    "C:/Users/ebin5/ENEE408I_Files/2025_Spring_Team_3/terpiez-project-ebin-6a5faac731b4.json"
)
client = speech.SpeechClient(credentials=creds)

# === AUDIO STREAM SETUP ===
RATE = 16000
CHUNK = int(RATE / 10)

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

# === ROBOT MOTION FUNCTIONS ===
def move_command(pub, direction):
    twist = Twist()
    duration = 1.5  # seconds

    if direction == "forward":
        twist.linear.x = 0.3
    elif direction == "backward":
        twist.linear.x = -0.3
    elif direction == "left":
        twist.angular.z = 1.57  # ~90 deg turn left
        duration = 1.0
    elif direction == "right":
        twist.angular.z = -1.57  # ~90 deg turn right
        duration = 1.0
    else:
        return

    print(f"🦾 Executing command: {direction}")
    pub.publish(twist)
    time.sleep(duration)
    pub.publish(Twist())  # stop

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
        config=config, interim_results=True
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
        for response in responses:
            if not response.results or not response.results[0].alternatives:
                continue

            transcript = response.results[0].alternatives[0].transcript.lower()
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

# === MAIN ===
if __name__ == "__main__":
    print("🚀 Starting Simon Says Voice Control")

    rospy.init_node('simon_says_teleop')
    pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)

    try:
        listen_and_execute(pub)
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        pub.publish(Twist())  # stop robot on exit
