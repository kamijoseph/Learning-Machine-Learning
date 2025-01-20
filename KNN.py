
# Simple K-Nearest Neighbours algorithm
import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k=3):
        self.k = k
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def predict(self, X):
        predictions = [self.predictor(x) for x in X]
        return np.array(predictions)
    
    def predictor(self, x):
        distance = [np.linalg.norm(x-x_train) for x_train in self.X_train]
        k_indices = np.argsort(distance)[:self.k]
        k_nearestLabels = [self.y_train[i] for i in k_indices]
        mostCommon = Counter(k_nearestLabels).most_common(1)
        return mostCommon[0][0]