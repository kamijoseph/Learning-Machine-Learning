
# Simple Support Vector Macine Classifier for 2-Dimensional Data from Scratch.
import numpy as np
import matplotlib.pyplot as plt

class SupportVectorMachine:
    def __init__(self, learningRate=0.001, lambdaParameter=0.01, n_iterations=1000):
        self.learningRate = learningRate
        self.lambdaParameter = lambdaParameter
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # labels --> +1 and -1
        y_ = np.where(y <= 0, -1, 1)
        
        for _ in range(self.n_iterations):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.weights) - self.bias) >= 1
                
                # point correctly classified ?
                if condition:
                    self.weights = self.learningRate * (2 * self.lambdaParameter * self.weights)
                # incorrectly classified ?
                else:
                    self.weights -= self.learningRate * (2 * self.lambdaParameter * self.weights - np.dot(x_i, y_[idx]))
    
    def predict(self, X):
        approximate = np.dot(X, self.weights) - self.bias
        return np.sign(approximate)