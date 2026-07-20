from typing import List, Union

import gpiod
from gpiod.line import Direction, Value
import serial

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
    """Control one laser-mask GPIO line with libgpiod v2."""

    def __init__(
        self,
        chip: Union[int, str] = 0,
        line: int = 2,
        *,
        active_low: bool = False,
    ) -> None:
        if isinstance(chip, int):
            if chip < 0:
                raise ValueError("chip must be non-negative")
            chip = f"/dev/gpiochip{chip}"
        if line < 0:
            raise ValueError("line must be non-negative")

        self.chip = chip
        self.line = line
        self._request = gpiod.request_lines(
            chip,
            consumer="laser-mask",
            config={
                line: gpiod.LineSettings(
                    direction=Direction.OUTPUT,
                    active_low=active_low,
                    output_value=Value.ACTIVE,
                )
            },
        )

    def _gpio(self):
        if self._request is None:
            raise RuntimeError("LaserMask is closed")
        return self._request

    def state(self) -> bool:
        return self._gpio().get_value(self.line) is Value.ACTIVE

    @property
    def is_on(self) -> bool:
        return self.state()

    def set(self, enabled: bool) -> None:
        value = Value.ACTIVE if enabled else Value.INACTIVE
        self._gpio().set_value(self.line, value)

    def on(self) -> None:
        self.set(True)

    def off(self) -> None:
        self.set(False)

    def toggle(self) -> bool:
        new_state = not self.state()
        self.set(new_state)
        return new_state

    def close(self) -> None:
        request = self._request
        if request is None:
            return
        try:
            request.set_value(self.line, Value.INACTIVE)
        finally:
            request.release()
            self._request = None

    def __enter__(self) -> "LaserMask":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
