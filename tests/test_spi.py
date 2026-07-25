import time
from periphery import SPI

spi = SPI(
    devpath="/dev/spidev0.0",
    mode=0,
    max_speed=32_000_000,
)

# 当前下位机协议要求固定发送16字节
tx_data = bytearray(16)
tx_data[0:5] = b"hello"

try:
    while True:
        rx_data = bytes(spi.transfer(tx_data))

        print("TX:", tx_data.hex(" "))
        print("RX:", rx_data.hex(" "))

        # 当前下位机在主循环中重新装载DMA，需要帧间隔
        time.sleep(0.001)
finally:
    spi.close()
