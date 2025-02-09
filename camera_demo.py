"""
A simple script to test the webcam and determine its maximum FPS.
Test environment result: ~30 FPS.
"""

import time

import cv2

# Open the default webcam (device 0)
cap = cv2.VideoCapture(0)

# Check if the webcam is opened successfully
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

frame_times = []

while cv2.waitKey(1) != ord('q'):  # Up to 1000 loops per second
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:  # If frame read fails
        print("Failed to get frame")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    capture_time = time.time()

    # Display the frame
    cv2.imshow("Webcam", frame)

    current_time = time.time()
    while frame_times and frame_times[0] < current_time - 1:
        frame_times.pop(0)
    frame_times.append(capture_time)
    fps = len(frame_times)

    print("Delay", current_time - capture_time)
    print("FPS", fps)
    print()

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
