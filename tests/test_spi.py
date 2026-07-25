from periphery import SPI


spi = SPI(devpath="/dev/spidev0.0", mode=0, max_speed=32000000)
while True:
    print(spi.transfer(bytearray(b"hello")))
