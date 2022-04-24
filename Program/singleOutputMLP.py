import numpy as np
import DataGenerator as dg
import matplotlib.pyplot as plt

np.random.seed(314159265)



def sigmoid(x):
    return 1/(1+np.e**(-x))
def dSigmoid(x):
    return sigmoid(x) * (1-sigmoid(x))

def ReLU(x):
    return x * (x >= 0)
def dReLU(x):
    return 1 * (x >= 0)

def forward(inp, weights, bias):
    inp = np.array(inp)
    layerOuts = []
    for i, w in enumerate(weights):
        # reshape the weight matrix to match shape of input matrix
        # then take the dot product
        formattedWeights = (w[~np.isnan(w)].reshape(-1,len(inp)))
        #print(formattedWeights, inp, np.dot(formattedWeights, inp))
        if w in weights[-1]:
            inp = sigmoid(np.dot(formattedWeights, inp) + bias[i, ~np.isnan(bias[i])])
        else:
            inp = ReLU(np.dot(formattedWeights, inp) + bias[i, ~np.isnan(bias[i])])
        layerOuts.append(inp)
    return float(inp), layerOuts

if __name__ == "__main__":
    error = []
    
    weights = np.random.uniform(-1,1,(2,64))
    weights[1,32:] = np.nan
    
    bias = np.zeros((2,32))
    bias[1,1:] = np.nan
    
    epochs = 1000
    for i in range(epochs):
        n = 0.01 * (.99)**(i/epochs)
        inp, target = dg.getXOR()
        output, layerOuts = forward(inp, weights, bias)
                

        #correct weights
        weights[0].reshape(-1,2).T[0] -= n * 2 * (output-target) * dSigmoid(output) * weights[1, ~np.isnan(weights[1])] * dReLU(layerOuts[0]) * inp[0]
        weights[0].reshape(-1,2).T[1] -= n * 2 * (output-target) * dSigmoid(output) * weights[1, ~np.isnan(weights[1])] * dReLU(layerOuts[0]) * inp[1]
        weights[1, ~np.isnan(weights[1])] -= n * 2 * (output-target) * dSigmoid(output) * layerOuts[0]
        
        #Correct bias
        bias[0] -= n * -2 * (target-output) * dSigmoid(output) * weights[1, ~np.isnan(weights[1])] * dReLU(layerOuts[0])
        bias[1, ~np.isnan(bias[1])] -= n * -2 * (target-output) * dSigmoid(output)
    
        error.append((target-output)**2)
        print(error[-1])
        
    plt.plot(range(len(error)), error) 
    plt.show()
