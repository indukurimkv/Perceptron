import random


def getData():
    bits = [random.getrandbits(1) for i in range(2)]
    XOR = bits[0] ^ bits[1]
    return bits, XOR