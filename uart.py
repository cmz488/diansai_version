import time
import serial
from tools.peripheral import LaserMask, Uart
from tools.web import CameraManager
import cv2

if __name__ == "__main__":
    uart = Uart("/dev/tty2", baudrate=1500000)
    while True:
        uart.send(bytearray("hello", "utf-8"))
        print("开始发送")
        time.sleep(1)
