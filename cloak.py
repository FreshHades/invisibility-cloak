import cv2
import numpy as np

# Load background once
bg = None

def init_background(cap):
    global bg
    for i in range(30):
        _, bg = cap.read()
    bg = np.flip(bg, axis=1)

init_background(cv2.VideoCapture(0))

def apply_cloak_effect(frame):
    global bg
    frame = np.flip(frame, axis=1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Pink HSV Range
    lower_pink = np.array([135, 50, 50])
    upper_pink = np.array([175, 255, 255])

    mask = cv2.inRange(hsv, lower_pink, upper_pink)
    mask_inv = cv2.bitwise_not(mask)

    res1 = cv2.bitwise_and(bg, bg, mask=mask)
    res2 = cv2.bitwise_and(frame, frame, mask=mask_inv)
    final_output = cv2.addWeighted(res1, 1, res2, 1, 0)

    return final_output
