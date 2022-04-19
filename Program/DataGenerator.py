import random

index = 0
def getData():
    global index
    bits = [random.getrandbits(1) for i in range(2)]
    if index%4 == 0:
        bits = [0,0]
    if index%4 == 1:
        bits = [0,1]
    if index%4 == 2:
        bits = [1,0]
    if index%4 == 3:
        bits = [1,1]
    XOR = bits[0] ^ bits[1]
    
    AND = bits[0] and bits[1]
    
    index +=1
    
    return bits, AND