
# implementing a perceptron algoritm in python
import numpy as np

def unitStepFunction(x):
    return np.where(x > 0, 1, 0)


class Perceptron:
    def __init__(self, learningRate=0.01, nIterations=1000):
        self.learningRate = learningRate
        self.nIterations = nIterations
        self.activationFunction = unitStepFunction
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        nSamples, nFeatures = X.shape
        
        self.weights = np.zeros(nFeatures)
        self.bias = 0
    
    def predict(self, X):
        pass