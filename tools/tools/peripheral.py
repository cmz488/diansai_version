from typing import List

import serial
from periphery import GPIO

FRAME_HEAD = bytearray([0xFC, 0xCF])
FRAME_TAIL = bytearray([0xCF, 0xFC])


class Uart:
    def __init__(self, port: str = "/dev/ttyS2", baudrate: int = 230400):
        self.ser = serial.Serial(port=port, baudrate=baudrate)

    def send(self, data: bytearray):
        self.ser.write(FRAME_HEAD + data + FRAME_TAIL)

    def read(self) -> List:
        return list(self.ser.read(self.ser.in_waiting))


class LaserMask:
    def __init__(self, port: int = 98) -> None:
        self.gpio = GPIO(port, "out")
        self.gpio.write(False)
        self.gpio_status: bool = False

    def on(self):
        self.gpio_status = True
        self.gpio.write(True)

    def off(self):
        self.gpio_status = False
        self.gpio.write(False)

    def toggle(self):
        if self.gpio_status:
            self.off()
        else:
            self.on()
