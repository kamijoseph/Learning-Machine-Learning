
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
                
    
    def regionQuery(self):
        pass
    
    def expandCluster(self):
        pass
    
    def predict(self):
        pass