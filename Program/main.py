import DataGenerator as DG
import SkeletonNueralNet as SNN
import numpy as np

def forward(input, weights):
    for i in weights:
        input = np.dot(input, i)
        
    return input

if __name__ == "__main__":
    netIn, target = DG.getData()
    weights = SNN.genSNN(2,1,1)
    netOut = forward(netIn, weights)
    