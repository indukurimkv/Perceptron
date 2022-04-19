import DataGenerator as DG
import SkeletonNueralNet as SNN
import numpy as np
import time

def sigDer(x):
    return 1/(1+np.e**(-x)) * (1-1/(1+np.e**(-x)))
def forward(inputs, weights):
    layerOuts = []
    for i in weights:
        inputs = np.dot(inputs, i)
        inputs = 1/(1+np.e**(-inputs))
        layerOuts.append(inputs)
        
    return inputs, layerOuts

if __name__ == "__main__":
    n = 0.1
    
    weights = SNN.genSNN(2,4,2)

    for i in range(1000000):
        netIn, target = DG.getData()
        netOut, layerOuts = forward(netIn, weights)
        delta0 = netOut*(1-netOut)*(target - netOut)
        weights[2][0] += n * (target - netOut) * (1 - netOut) * sigDer(netOut) * netOut
        weights[1][0] += n * (target - netOut) * (1 - netOut) * sigDer(netOut) * sigDer(float(layerOuts[1])) * float(layerOuts[1])
        weights[0][0] += n * (target - netOut) * (1 - netOut) * sigDer(netOut) * sigDer(float(layerOuts[1])) * sigDer(float(layerOuts[0])) * netIn[1] 
        weights[0][1] += n * (target - netOut) * (1 - netOut) * sigDer(netOut) * sigDer(float(layerOuts[1])) * sigDer(float(layerOuts[0])) * netIn[0]
        if(np.square(target-netOut) < 0.15):
            n = 0.01
        print(np.square(target-netOut))
    