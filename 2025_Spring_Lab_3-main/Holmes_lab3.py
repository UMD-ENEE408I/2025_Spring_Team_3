import cv2
import numpy as np

def detectLine(frame):
    """
    Process the given frame to detect and track the center of a white line.
    
    Args:
        frame (numpy.ndarray): The input frame from the webcam.
    
    Returns:
        lineCenter: A number between [-1, 1] denoting where the center of the line is relative to the frame.
        newFrame: Processed frame with the detected line marked using cv2.rectangle() and center marked using cv2.circle().
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Threshold to detect white color (adjust thresholds as needed)
    _, threshold = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY)
    
    # Find contours of the white areas
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    height, width = frame.shape[:2]
    line_center_x = width // 2  # Default to the center if no line is detected

    if contours:
        # Find the largest contour, assuming it's the line
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)

        if M['m00'] != 0:
            # Calculate the center of the line
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            # Draw the contour and center point on the frame
            cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 3)
            cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

            # Calculate the normalized position of the line center relative to the frame
            line_center_x = cx

    # Normalize lineCenter between -1 and 1
    lineCenter = (2 * (line_center_x / width)) - 1

    return lineCenter, frame


def main():
    cam = cv2.VideoCapture(0)  # Open webcam

    if not cam.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        lineCenter, newFrame = detectLine(frame)

        # Display line position in console (for debugging)
        print(f"Line Center: {lineCenter:.2f}")

        cv2.imshow('Line Tracking', newFrame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
