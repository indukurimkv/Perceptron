import numpy as np

np.random.seed(69420)

def genSNN(numInps, *shape):
    weightShape = []
    
    #Get the shape for the weight matrix
    _prevLayer = numInps
    for i, length in enumerate(shape):
        weightShape.append(_prevLayer*length)
        _prevLayer = length
    
    #make a weight matrix
    weightMatrix = np.random.uniform(-1,1, (len(weightShape), max(weightShape)))
    
    #Remove unnecessary weights
    for i, layerShape in enumerate(weightShape):
        weightMatrix[i, layerShape:] = np.NAN
    
    biasMatrix = np.zeros((len(shape), max(shape)))
    for i, layerShape in enumerate(shape):
        biasMatrix[i, layerShape:] = np.NAN
    
    return weightMatrix, biasMatrix

def getSNNAsMatrix(numInps, *shape):
    netShape = (numInps, *shape)
    weights, biases = genSNN(numInps, *shape)
    return np.matrix(weights), np.matrix(biases)