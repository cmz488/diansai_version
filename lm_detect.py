import time

import cv2
import numpy as np


def main():
    cap = cv2.VideoCapture(1, cv2.v4l2)
    last = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            raise "not found"
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        current = time.time()
        fps = 1 / (current - last)
        last = current
        print(fps)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"退出,throw {e}")
