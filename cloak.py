import cv2
import numpy as np

bg = None  # Background frame (global)

def init_background(cap):
    global bg
    for i in range(30):
        success, frame = cap.read()
        if success and frame is not None:
            bg = np.flip(frame, axis=1)
    if bg is None:
        raise ValueError("Background initialization failed: Could not capture from webcam.")

def apply_cloak_effect(frame):
    global bg
    if frame is None or bg is None:
        return frame

    frame = np.flip(frame, axis=1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Pink HSV range
    lower_pink = np.array([140, 50, 50])
    upper_pink = np.array([170, 255, 255])
    mask = cv2.inRange(hsv, lower_pink, upper_pink)

    mask_inv = cv2.bitwise_not(mask)

    cloak_area = cv2.bitwise_and(bg, bg, mask=mask)
    normal_area = cv2.bitwise_and(frame, frame, mask=mask_inv)

    final_output = cv2.addWeighted(cloak_area, 1, normal_area, 1, 0)
    return final_output

