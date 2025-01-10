
# Linear Regression Model Implementation From Scratch
import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        weights = None
        bias = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = 0
        self.bias = 0
        
        for _ in range(n_iterations=1000):
            y_predict = np.dot(X, self.weights) + self.bias
            
            dw = (1/n_samples) * np.dot(X.T (y_predict-y))
            db = (1/n_samples) * np.sum(y_predict-y)
            
            # Update Weights and Bias
            self.weights = self.weights - self.learning_rate * dw
            self.bias = self.bias - self.learning_rate * db
    
    def predict(self, X):
        pass