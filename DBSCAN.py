
# Implementation of Density-Based Spatial Clustering of Applications with Noise (DBSCAN) algorithm
from collections import deque
import numpy as np
import matplotlib.pyplot as plt

class DBSCAN:
    def __init__(self, eps=0.5, minSamples=5):
        self.eps = eps
        self.minSamples = minSamples
        self.labels = None
    
    def fit(self):
        pass
    
    def regionQuery(self):
        pass
    
    def expandCluster(self):
        pass
    
    def predict(self):
        pass