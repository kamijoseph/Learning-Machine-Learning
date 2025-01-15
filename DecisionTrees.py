
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
    
    def gini(self):
        pass
    
    def split(self):
        pass
    
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