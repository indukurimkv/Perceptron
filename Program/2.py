import numpy as np
import SkeletonNueralNet as snn
import DataGenerator as dg
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1+np.e**(-x))
def dSigmoid(x):
    return sigmoid(x) * (1-sigmoid(x))

def ReLU(x):
    return x * (x >= 0)
def dReLU(x):
    return 1 * (x >= 0)

def meanSqErr(target, prediction):
    return ((target-output)**2).mean()
def dMeanSqErr(target, prediction):
    return np.matrix(2 * 1/len(prediction) * (prediction-target))

def forward(inp, weights, bias):
    inp = np.array(inp)
    layerOuts = []
    for i, w in enumerate(weights):
        # reshape the weight matrix to match shape of input matrix
        # then take the dot product
        formattedWeights = np.split(w.A1[~np.isnan(w.A1)], len(inp))
        if i == len(weights)-1:
            inp = sigmoid(np.dot(inp, formattedWeights) + bias[i, ~np.isnan(bias[i])])
        else:
            inp = ReLU(np.dot(inp, formattedWeights) + bias[i, ~np.isnan(bias[i])])
        layerOuts.append(inp)
    return inp, layerOuts

if __name__ == "__main__":
    netShape = (2,3,2)
    weights, biases = snn.genSNN(*netShape)
    
    n = 0.001
    epochs = 10000
    errors = []
    for epoch in range(epochs):
        
        dataIn, target = np.array(dg.doubleOutAnd())
        output, layerOuts = forward(dataIn, weights, biases)
        layerOuts = layerOuts[::-1]
        dError = dMeanSqErr(target, output)
        
        outputNodes = np.split(weights.A[-1], netShape[-1])
            
        biases[0]-= ((((dError.T * np.matrix(dSigmoid(output))).diagonal() * weights[-1].reshape(-1,netShape[1])).T * dReLU(layerOuts[-1])).diagonal() * n).A1
        biases[-1, ~np.isnan(biases[-1])] -= ((dError.T * dSigmoid(output)).diagonal() * n).A1

        weights[0] -= (np.matrix(dataIn).T * (((dError.T * np.matrix(dSigmoid(output))).diagonal() * weights[-1].reshape(-1,netShape[1])).T * dReLU(layerOuts[-1])).diagonal()).A1 * n
        weights[-1] -= ((dError.T * dSigmoid(output)).diagonal().T * layerOuts[-1]).T.A1 * n
        
        
        #Tracking info
        currentError = np.average(meanSqErr(dataIn, output))
        print(currentError) if epoch%100 == 0 else None
        errors.append(currentError)

    plt.plot(range(len(errors)), errors)
    plt.show()
