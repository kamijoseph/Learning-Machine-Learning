
# implementing a perceptron algoritm in python
import numpy as np

def unitStepFunction(x):
    return np.where(x > 0, 1, 0)


class Perceptron:
    def __init__(self):
        pass
    
    def fit(self, X, y):
        pass
    
    def predict(self, X):
        pass