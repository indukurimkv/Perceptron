"""
Architecture 2,2,1
In: shape:(2,)

Out: shape: (2,1) then (2,1) then (1,)
"""
import numpy as np
import sympy as sp
from sympy.matrices import Matrix

np.random.seed(0)
class Node:
    def __init__(self, layerIndex, numWeights):
        self.layerIndex = layerIndex
        self.keyedWeights = {}
        self.weights = []
        self.weightSymbols = []
        for i in range(numWeights):
            _ = np.random.uniform(-1,1)
            self.keyedWeights["{},{}".format(layerIndex, i)] = _
            self.weights.append(_)
            self.weightSymbols.append(sp.symbols("{}|{}".format(layerIndex,i)))
        self.weightSymbols = Matrix(self.weightSymbols)
        self.weights = Matrix(self.weights)
        
    def forward(self, inp):
        self.input = Matrix(inp)
        self.simOutput = sigmoid(self.weightSymbols.dot(self.input))
        self.output = self.simOutput
        for i,_ in enumerate(self.weights):
            self.output = self.output.subs(self.weightSymbols[i], self.keyedWeights["{},{}".format(self.layerIndex,i)])
        return self.output
    
    def backProp(self, partialDer: str):
        pass
    
    def __repr__(self):
        return str(self.weights)
    def __str__(self):
        return str(self.weights)
    def __getitem__(self, i):
        return self.weights[i]
    
    
class Layer:
    def __init__(self, index, numNodes, numPreviousNodes):
        self.nodes = [Node(index, numPreviousNodes) for i in range(numNodes)]
        self.index = index
        
    def forward(self, inp):
        return [n.forward(inp) for n in self.nodes]
        
    def __repr__(self):
        return str([n for n in self.nodes])
        
    def __getitem__(self, i):
        return self.nodes[i]
def sigmoid(x):
    return 1/(1+sp.exp(-x))
def dSigmoid(x):
    return sigmoid(x) * (1-sigmoid(x))


if __name__ == "__main__":
    layer = Layer(0,2,2)
