import time
import threading
import cv2
import numpy as np
import serial
from typing import Union
import serial
from typing import Optional, Union

# 定义帧头/帧尾（可根据需要修改）
FRAME_HEAD = b"\xaa"
FRAME_TAIL = b"\xbb"


# 全局标志，用于控制线程退出
class Uart:
    def __init__(
        self,
        port: str = "/dev/ttyS2",
        baudrate: int = 230400,
        timeout: float = 0.5,
        add_frame: bool = False,
    ):
        """
        :param add_frame: 是否自动添加帧头/帧尾（默认不添加，便于回环测试）
        """
        self.add_frame = add_frame
        self.ser = serial.Serial(port=port, baudrate=baudrate, timeout=timeout)

    def send(self, data: Union[bytes, bytearray]) -> int:
        """发送数据，返回实际写入字节数"""
        payload = bytearray(data)
        if self.add_frame:
            payload = FRAME_HEAD + payload + FRAME_TAIL
        try:
            return self.ser.write(payload)
        except serial.SerialException as e:
            print(f"[UART] 发送失败: {e}")
            return 0

    def read(self, size: Optional[int] = None) -> bytes:
        """
        读取最多 size 个字节，若 size 为 None 则读取当前缓冲区所有数据。
        返回 bytes 对象，便于后续处理。
        """
        if size is None:
            size = self.ser.in_waiting
        try:
            return self.ser.read(size)
        except serial.SerialException as e:
            print(f"[UART] 读取失败: {e}")
            return b""


# --------------------- Uart 类（改进版，支持关闭帧头帧尾） ---------------------
class Uart:
    def __init__(
        self,
        port: str = "/dev/ttyTHS1",
        baudrate: int = 230400,
        timeout: float = 0.5,
        add_frame: bool = False,
    ):
        """
        串口封装类
        :param port:      串口设备路径
        :param baudrate:  波特率
        :param timeout:   读取超时（秒）
        :param add_frame: 是否自动添加帧头 FRAME_HEAD 和帧尾 FRAME_TAIL
        """
        self.add_frame = add_frame
        self.ser = serial.Serial(port=port, baudrate=baudrate, timeout=timeout)

    def send(self, data: Union[bytes, bytearray]) -> int:
        """发送数据，返回实际发送字节数"""
        payload = bytearray(data)
        if self.add_frame:
            # 若需要帧封装，请在此定义 FRAME_HEAD 和 FRAME_TAIL
            FRAME_HEAD = b"\xaa"
            FRAME_TAIL = b"\xbb"
            payload = FRAME_HEAD + payload + FRAME_TAIL
        try:
            return self.ser.write(payload)
        except serial.SerialException as e:
            print(f"[UART] 发送失败: {e}")
            return 0

    def read(self, size: Optional[int] = None) -> bytes:
        """读取最多 size 字节，若 size 为 None 则读取当前缓冲区全部数据"""
        if size is None:
            size = self.ser.in_waiting
        try:
            return self.ser.read(size)
        except serial.SerialException as e:
            print(f"[UART] 读取失败: {e}")
            return b""

    def close(self):
        """关闭串口"""
        if self.ser.is_open:
            self.ser.close()


# --------------------- 全局运行标志 ---------------------
running = True


# --------------------- 发送线程 ---------------------
def sender(uart: Uart):
    """每隔 0.5 秒发送一次 "hello" """
    while running:
        try:
            n = uart.send(bytearray("hello".encode("utf-8")))
            print(f"[发送] 发送了 {n} 字节: hello")
        except Exception as e:
            print(f"[发送] 异常: {e}")
        time.sleep(0.5)


# --------------------- 接收线程 ---------------------
def receiver(uart: Uart):
    """持续读取串口数据并打印（十六进制和字符串形式）"""
    while running:
        try:
            # 检查缓冲区是否有数据
            available = uart.ser.in_waiting
            if available > 0:
                data = uart.read(available)
                if data:
                    # 打印十六进制和可打印字符（忽略解码错误）
                    hex_str = data.hex()
                    try:
                        text = data.decode("utf-8")
                    except UnicodeDecodeError:
                        text = repr(data)
                    print(f"[接收] {len(data)} 字节: {hex_str}  ->  {text}")
        except Exception as e:
            print(f"[接收] 异常: {e}")
        time.sleep(0.01)  # 避免空转占用过高 CPU


# --------------------- 主程序 ---------------------
if __name__ == "__main__":
    # ========== 1. 配置串口 ==========
    # 请根据实际设备修改端口号，例如 "/dev/ttyTHS1" 或 "/dev/ttyS2"
    PORT = "/dev/ttyTHS1"
    BAUDRATE = 230400
    TIMEOUT = 0.5
    ADD_FRAME = False  # 回环测试时建议关闭帧封装，直接收发原始数据

    print(f"正在打开串口 {PORT}，波特率 {BAUDRATE} ...")
    uart = Uart(port=PORT, baudrate=BAUDRATE, timeout=TIMEOUT, add_frame=ADD_FRAME)
    print("串口已打开！")

    # ========== 2. 硬件回环提示 ==========
    print("\n【重要】请确保已将串口的 TX 和 RX 引脚短接（硬件回环）！")
    print("如果未短接，接收线程将收不到数据，测试将无法验证。\n")

    # ========== 3. 启动收发线程 ==========
    t_send = threading.Thread(target=sender, args=(uart,), name="SenderThread")
    t_recv = threading.Thread(target=receiver, args=(uart,), name="ReceiverThread")
    t_send.start()
    t_recv.start()

    # ========== 4. 主线程等待退出 ==========
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n用户按下 Ctrl+C，正在退出...")
        running = False
        t_send.join(timeout=1)
        t_recv.join(timeout=1)
        uart.close()
        print("已关闭串口，程序退出。")
