
# Decision trees algorithm from scratch
import numpy as np
from collections import Counter

class DecisionTrees:
    def __init__(self, maxDepth=None):
        self.maxDepth = maxDepth
        self.tree = None
    
    class Node:
        def __init__(self, feature=None, threshhold=None, left=None, right=None):
            self.feature = feature
            self.threshhold = threshhold
            self.left = left
            self.right = right
    
    # Genie impurity function, haha
    def gini(self, y):
        classCount = Counter(y)
        n_samples = len(y)
        gini = 1.0 - sum((count / n_samples) ** 2 for count in classCount.values())
        return gini
    
    def split(self, X, y, feature, threshhold):
        leftIdxs = np.where(X[:, feature] <= threshhold)[0]
        rightIdxs = np.where(X[:, feature] > threshhold)[0]
        return leftIdxs, rightIdxs
    
    def bestSplit(self):
        pass
    
    def buildTree(self):
        pass
    
    def fit(self):
        pass
    
    def traverseTree(self):
        pass
    
    def predict(self):
        pass