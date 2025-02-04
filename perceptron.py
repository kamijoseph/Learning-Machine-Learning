
# implementing a perceptron algoritm in python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

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
        y_ = np.where(y > 0, 1, 0)
        
        # learning weights
        for _ in range(self.nIterations):
            for idx, xi in enumerate(X):
                linearOutput = np.dot(xi, self.weights) + self.bias
                yPredict = self.activationFunction(linearOutput)
                
                # perceptron update
                update = self.learningRate * (y_[idx] - yPredict)
                self.weights += update * xi
                self.bias += update
        
        
    def predict(self, X):
            linearOutput = np.dot(X, self.weights) + self.bias
            yPredict = self.activationFunction(linearOutput)
            return yPredict

if __name__ == "__main__":
    pass