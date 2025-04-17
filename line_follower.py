import cv2
import numpy as np
import time

# === PID Controller ===
class PID:
    def __init__(self, kp=0.005, ki=0.0001, kd=0.001):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0
        self.integral = 0
        self.last_time = time.time()

    def compute(self, error):
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time

        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        self.prev_error = error

        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        return output

# === Setup Camera ===
cap = cv2.VideoCapture(0)
cap.set(3, 160)  # width
cap.set(4, 120)  # height

pid = PID()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        height, width, _ = frame.shape

        # Focus on bottom section of the frame
        roi = frame[int(height * 0.75):, :]

        # Convert to grayscale and threshold to detect white line
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])  # centroid x
                cy = int(M["m01"] / M["m00"])  # centroid y

                # Compute error from center
                error = cx - width // 2
                correction = pid.compute(error)

                print(f"[INFO] Centroid: {cx}, Error: {error}, PID Correction: {correction:.3f}")

                # Visualization
                cv2.drawContours(roi, [largest], -1, (0, 255, 0), 2)
                cv2.circle(roi, (cx, cy), 5, (0, 0, 255), -1)
        else:
            print("[WARN] No line detected.")

        # Show visual debug
        cv2.imshow("Line Follower View", roi)
        cv2.imshow("Binary Mask", binary)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()