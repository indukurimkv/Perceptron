import DataGenerator as DG
import SkeletonNueralNet as SNN
import numpy as np
import time

def forward(inputs, weights):
    layerOuts = []
    for i in weights:
        inputs = np.dot(inputs, i)
        layerOuts.append(inputs)
        
    return inputs, layerOuts

if __name__ == "__main__":
    n = 0.9
    weights = SNN.genSNN(2,1)

    for i in range(50):
        netIn, target = DG.getData()
        netOut, layerOuts = forward(netIn, weights)
        delta0 = netOut*(1-netOut)*(target - netOut)
        weights[-1]  += n * delta0 * layerOuts[0]
        weights[0][0] +=n * netIn[1] * layerOuts[0] * (1-layerOuts[0]) * delta0
        weights[0][1] +=n * netIn[0] * layerOuts[0] * (1-layerOuts[0]) * delta0
        print(weights)
        print(np.square(target-netOut))
    