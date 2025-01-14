
# Simple Support Vector Macine Classifier for 2-Dimensional Data from Scratch.
import numpy
import matplotlib.pyplot as plt

class SupportVectorMachine:
    def __init__(self, learningRate=0.001, lambdaParameter=0.01, n_iterations =1000):
        self.learningRate = learningRate
        self.lambdaParameter = lambdaParameter
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        
    def fit(self):
        pass
    
    def predict(self):
        pass