
# K Means Clustering Algorithm
import numpy as np
import matplotlib.pyplot as plt

class KMeans:
    def __init__(self, n_clusters=3, maxIterations=100, tolerance=1e-4):
        self.n_clusters = n_clusters
        self.maxIterations = maxIterations
        self.tolerance = tolerance
        self.centroids = None
    
    def initializeCentroids(self, X):
        indices = np.random.choice(self.X.shape[0], self.n_clusters, replace=False)
        return X[indices]
    
    def calculateDistance(self):
        pass
    
    def fit(self):
        pass
    
    def predict(self):
        pass