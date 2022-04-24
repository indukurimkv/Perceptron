import random

def getAND():
    bits = [random.getrandbits(1) for i in range(2)]
    AND = bits[0] and bits[1]
    return bits, AND

def getXOR():
    bits = [random.getrandbits(1) for i in range(2)]
    XOR = bits[0] ^ bits[1]
    return bits, XOR

def doubleOutAnd():
    bits, result = getAND()
    result = [0,1] if not result else [1,0]
    return bits, result