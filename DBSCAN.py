
# Implementation of Density-Based Spatial Clustering of Applications with Noise (DBSCAN) algorithm
from collections import deque
import numpy as np
import matplotlib.pyplot as plt

class DBSCAN:
    def __init__(self, eps=0.5, minSamples=5):
        self.eps = eps
        self.minSamples = minSamples
        self.labels = None
    
    def fit(self, X):
        n_samples = X.shape[0]
        self.labels = -np.ones(n_samples)
        clusterId = 0
        
        for i in range(n_samples):
            if self.labels[i] != -1:
                continue
            
            neighbors = self.regionQuery(X, i)
            if len(neighbors) < self.minSamples:
                self.labels[i] = -1
            else:
                self.expandCluster(X, i, neighbors, clusterId)
                clusterId += 1
    
    def regionQuery(self, X, pointIdx):
        distances = np.linalg.norm(X - X[pointIdx], axis=1)
        return np.where(distances <= self.eps)[0]
    
    def expandCluster(self, X, pointIdx, neighbors, clusterId):
        self.labels[pointIdx] = clusterId
        queue = deque(neighbors)
        while queue:
            neighborIdx = queue.popleft()
            
            if self.labels[neighborIdx] == -1:
                self.labels[neighborIdx] = clusterId
            
            if self.labels[neighborIdx] != -1:
                continue
            self.labels[neighborIdx] = clusterId
            
            newNeighbours = self.regionQuery(X, neighborIdx)
            if len(newNeighbours) >= self.minSamples:
                queue.extend(newNeighbours)
    
    def predict(self):
        return self.labels
    
