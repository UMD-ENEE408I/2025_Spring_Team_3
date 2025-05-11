#!/usr/bin/env python3

# THIS CODE NOW WORKS FOR GESTURE DETECTION AND SIMON SAYS 
# LEZZZZZZZZZZZ GGGGGOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
import rospy
import cv2
import re
import threading
import queue
import torch
from ultralytics import YOLO
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
from google.cloud import speech
from google.oauth2 import service_account
import pyaudio

# === Constants ===
RATE = 16000
CHUNK = int(RATE / 10)

# === Load YOLOv11 model onto GPU and confirm device ===
model = YOLO("simonsaysv1.pt")
model.to("cuda")
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Model on GPU device:", next(model.model.parameters()).device)

# === ROS + CV Bridge Setup ===
subscriber_node_name = "camera_gesture_and_audio_node"
topic_name = "video_topic/compressed"
bridge = CvBridge()
frame_queue = queue.Queue(maxsize=2)

# === Shared State ===
simon_mode = False
state_lock = threading.Lock()


# === Movement stubs ===
def move_forward():
    rospy.loginfo("⬆️ Moving forward")


def move_backward():
    rospy.loginfo("⬇️ Moving backward")


def turn_left():
    rospy.loginfo("⬅️ Turning left")


def turn_right():
    rospy.loginfo("➡️ Turning right")


def stop_movement():
    rospy.loginfo("⏹️ Stopping movement")


# === Image callback: enqueue frames ===
def image_callback(msg):
    try:
        frame = bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
        if frame is None:
            rospy.logwarn("⚠️ Frame conversion returned None")
            return
        resized = cv2.resize(frame, (640, 480))
        if not frame_queue.full():
            frame_queue.put(resized)
    except Exception as e:
        rospy.logerr(f"❌ Error converting frame: {e}")


# === Processing Thread: run YOLO & fire moves only when in Simon mode ===
def process_frames():
    global simon_mode
    while not rospy.is_shutdown():
        try:
            frame = frame_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        # snapshot current simon_mode
        with state_lock:
            sm = simon_mode

        # run inference, suppress prints when not in Simon mode
        res = model(frame, verbose=sm)  # verbose=True only if sm==True
        results = res[0]
        annotated = results.plot()

        # if Simon mode, trigger the first detected gesture action
        if sm:
            with state_lock:
                for box in results.boxes:
                    label = model.names[int(box.cls)]
                    if label == "forward":
                        move_forward()
                        simon_mode = False
                        break
                    if label == "backward":
                        move_backward()
                        simon_mode = False
                        break
                    if label == "left":
                        turn_left()
                        simon_mode = False
                        break
                    if label == "right":
                        turn_right()
                        simon_mode = False
                        break
                    if label == "stop":
                        stop_movement()
                        simon_mode = False
                        break

        # always show camera feed
        cv2.imshow("YOLOv11 Gesture", annotated)
        cv2.waitKey(1)


# === Audio → “Simon says” detection ===
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
    for response in responses:
        if not response.results or not response.results[0].alternatives:
            continue

        result = response.results[0]
        transcript = result.alternatives[0].transcript
        is_final = result.is_final

        if is_final:
            rospy.loginfo(f"🗣️ Heard: {transcript}")
            if re.search(r"\bsimon says\b", transcript, re.I):
                with state_lock:
                    simon_mode = True
                rospy.loginfo("🗣️ Simon says! Next gesture will move.")


def audio_thread_fn():
    creds = service_account.Credentials.from_service_account_file("key.json")
    client = speech.SpeechClient(credentials=creds)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code="en-US",
    )
    streaming_config = speech.StreamingRecognitionConfig(
        config=config, interim_results=True
    )

    with MicrophoneStream() as stream:
        audio_gen = stream.generator()
        requests = (
            speech.StreamingRecognizeRequest(audio_content=chunk) for chunk in audio_gen
        )
        responses = client.streaming_recognize(streaming_config, requests)
        listen_print_loop(responses)


# === Main ===
if __name__ == "__main__":
    rospy.init_node(subscriber_node_name, anonymous=True)
    rospy.Subscriber(topic_name, CompressedImage, image_callback, queue_size=1)

    threading.Thread(target=process_frames, daemon=True).start()
    threading.Thread(target=audio_thread_fn, daemon=True).start()

    rospy.loginfo("🚀 Node running: gesture + Simon says audio control")
    rospy.spin()
    cv2.destroyAllWindows()
