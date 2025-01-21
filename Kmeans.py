
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
    
    def calculateDistance(self, X, centroids):
        distances = np.sqrt(((X[:, np.newaxis, :] - centroids) ** 2).sum(axis=2))
        return distances
    
    def fit(self, X):
        self.centroids = self.initializeCentroids(X)
        for iters in range(self.maxIterations):
            # assign points to the nearest centroid
            distances = self.calculateDistance(X, self.centroids)
            clusters = np.argmin(distances, axis=1)
            # update centroids
            newCentroids = np.array([X[clusters == k].mean(axis=0) if np.any(clusters == k) else self.centroids[k]
                                     for k in range(self.n_clusters)])
            if np.all(np.abs(newCentroids - self.centroids) < self.tolerance):
                break
            self.centroids = newCentroids
        self.clusters = clusters
    def predict(self):
        pass