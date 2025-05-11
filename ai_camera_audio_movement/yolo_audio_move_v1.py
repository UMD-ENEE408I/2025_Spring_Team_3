#!/usr/bin/env python3

# This code is at a good point, very nice. Move to version 2
# THIS CODE DOES NOT DO YOLO OR MOVEMENT

import rospy
from sensor_msgs.msg import CompressedImage
import cv2
from cv_bridge import CvBrAidge
import threading
import queue
import re
import sys
from google.cloud import speech
from google.oauth2 import service_account
import pyaudio

# === ROS Subscriber Setup ===
subscriber_node_name = "camera_sensor_subscriber"
topic_name = "video_topic/compressed"
bridge = CvBridge()
_first_frame_logged = False


def image_callback(message):
    """Display incoming video frames; log only the first one."""
    global _first_frame_logged
    if not _first_frame_logged:
        rospy.loginfo("📷 Received first video frame")
        _first_frame_logged = True
    try:
        frame = bridge.compressed_imgmsg_to_cv2(message, "bgr8")
        cv2.imshow("camera", frame)
        cv2.waitKey(1)
    except Exception as e:
        rospy.logerr(f"Error decoding/displaying frame: {e}")


# === Movement command stubs ===
def move_forward():
    rospy.loginfo("⬆️ Moving forward")
    # TODO: call your forward-driving function here


def move_backward():
    rospy.loginfo("⬇️Moving backward")
    # TODO: call your backward-driving function here


def turn_left():
    rospy.loginfo("⬅️ Turning left") 
    # TODO: call your left-turn function here


def turn_right():
    rospy.loginfo("➡️ Turning right")
    # TODO: call your right-turn function here


def stop_movement():
    rospy.loginfo("⏹️ Stopping movement")
    # TODO: call your stop function here


# === Audio Detection Setup ===
RATE = 16000
CHUNK = int(RATE / 10)
creds = service_account.Credentials.from_service_account_file("key.json")
speech_client = speech.SpeechClient(credentials=creds)
simon_mode = False


class MicrophoneStream:
    """Opens a recording stream as a generator yielding audio chunks."""

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
    """Handle streaming responses and look for Simon says + movement commands."""
    global simon_mode
    num_chars_printed = 0

    for response in responses:
        if not response.results:
            continue
        result = response.results[0]
        if not result.alternatives:
            continue

        transcript = result.alternatives[0].transcript
        overwrite = " " * max(0, num_chars_printed - len(transcript))

        if not result.is_final:
            # interim: overwrite in place
            sys.stdout.write(transcript + overwrite + "\r")
            sys.stdout.flush()
            num_chars_printed = len(transcript)
        else:
            # final transcript
            rospy.loginfo(f"🗣️ Heard: {transcript}")
            num_chars_printed = 0

            # If we're in Simon mode, check for movement commands
            if simon_mode:
                if re.search(r"\bforward\b", transcript, re.I):
                    move_forward()
                    simon_mode = False
                    continue
                if re.search(r"\bbackward\b", transcript, re.I):
                    move_backward()
                    simon_mode = False
                    continue
                if re.search(r"\bleft\b", transcript, re.I):
                    turn_left()
                    simon_mode = False
                    continue
                if re.search(r"\bright\b", transcript, re.I):
                    turn_right()
                    simon_mode = False
                    continue
                if re.search(r"\bstop\b", transcript, re.I):
                    stop_movement()
                    simon_mode = False
                    continue

            # Not in Simon mode (or command didn’t match): look for triggers
            if re.search(r"\bsimon says\b", transcript, re.I):
                rospy.loginfo("🗣️ 'Simon says' detected – awaiting command")
                simon_mode = True
                continue

            # Exit/quit still shuts down
            if re.search(r"\b(exit|quit)\b", transcript, re.I):
                rospy.loginfo("🗣️ Detected exit command, shutting down.")
                rospy.signal_shutdown("User requested exit")
                break


def audio_thread_fn():
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code="en-US",
    )
    streaming_config = speech.StreamingRecognitionConfig(
        config=config, interim_results=True
    )

    with MicrophoneStream(RATE, CHUNK) as stream:
        audio_gen = stream.generator()
        requests = (
            speech.StreamingRecognizeRequest(audio_content=chunk) for chunk in audio_gen
        )
        responses = speech_client.streaming_recognize(streaming_config, requests)
        listen_print_loop(responses)


def main():
    rospy.init_node(subscriber_node_name, anonymous=True)
    rospy.Subscriber(topic_name, CompressedImage, image_callback, queue_size=1)

    thread = threading.Thread(target=audio_thread_fn)
    thread.daemon = True
    thread.start()

    rospy.loginfo("🚀 Node running: video + Simon says control")
    rospy.spin()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
