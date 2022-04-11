import numpy as np

np.random.seed(69420)

def genSNN(*shape):
    net = []
    for i in shape:
        net.append(np.random.uniform(-1,1, i))
    return net