from periphery import GPIO


gpio = GPIO("/dev/gpiochip0", 43, "out")
while True:
    gpio.write(True)
