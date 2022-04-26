import numpy as np 
import DataGenerator as dg
import SkeletonNueralNet as SNN

def reshapeWeights(netShape, weight, layerNum):
    return weight.reshape(-1, netShape[layerNum+1])

if __name__ == "__main__":
    netShape = (2,3,2)
    weights, biases = SNN.getSNNAsMatrix(*netShape)