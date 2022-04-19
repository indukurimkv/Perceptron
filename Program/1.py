"""
Architecture 2,2,1
In: shape:(2,)

Out: shape: (2,1) then (2,1) then (1,)
"""
import numpy as np
import sympy as sp

np.random.seed(0)
class Node:
    def __init__(self, numWeights):
        self.weights = [np.random.uniform(-1,1) for i in range(numWeights)]
        
    def forward(self, inp):
        self.input = inp
        self.output = sigmoid(np.dot(self.weights, inp))
        return self.output
    
    def backProp(self, partialDer: str):
        self.derivative = eval(partialDer) * dSigmoid(self.output) * self.input
        #self.partialDer = eval(partialDer) * dSigmoid(self.output) * self.weights
        pass
    
    def __repr__(self):
        return str(self.weights)
    def __str__(self):
        return str(self.weights)
    def __getitem__(self, i):
        return self.weights[i]
    
    
class Layer:
    def __init__(self, numNodes, numPreviousNodes):
        self.nodes = [Node(numPreviousNodes) for i in range(numNodes)]
        
    def forward(self, inp):
        return [n.forward(inp) for n in self.nodes]
        
    def __repr__(self):
        return str([n for n in self.nodes])
        
    def __getitem__(self, i):
        return self.nodes[i]
def sigmoid(x):
    return 1/(1+np.e**(-x))
def dSigmoid(x):
    return sigmoid(x) * (1-sigmoid(x))


if __name__ == "__main__":
    layer = Layer(2,2)
