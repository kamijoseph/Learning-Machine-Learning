
# Decision trees algorithm from scratch
import numpy as np
from collections import Counter
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


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
    
    def bestSplit(self, X, y, feature, threshhold):
        n_samples, n_features = X.shape
        bestGini = float('inf')
        split = None
        
        for feature in range (n_features):
            threshholds = np.unique(X[:, feature])
            for threshold in threshholds:
                leftIdxs, rightIdxs = self.split(X, y, feature, threshhold)
                
                if len(leftIdxs) == 0 or len(rightIdxs) == 0:
                    continue
                
                leftGini = self.gini(y[leftIdxs])
                rightGini = self.gini(y[rightIdxs])
                
                weightedGini = (len(leftIdxs) * leftGini + len(rightIdxs) * rightGini) / n_samples
                if weightedGini < bestGini:
                    bestGini = weightedGini
                    split = {
                        'feature': feature,
                        'threshhold': threshhold,
                        'leftIdxs': leftIdxs,
                        'rightIdxs': rightIdxs
                    }
        return split
    
    # Build tree, yk recursively
    def buildTree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(set(y))
        if n_labels == 1 or depth == self.maxDepth or n_samples == 0:
            commonLabel = Counter(y).most_common(1)[0][0]
            return self.Node(value=commonLabel)
        
        split = self.bestSplit(X, y)
        if not split:
            commonLabel = Counter(y).most_common(1)[0][0]
            return self.Node(value=commonLabel)

        leftSubTree = self.buildTree(X[split['leftIdxs']], y[split['leftIdxs']], depth + 1)
        rightSubTree = self.buildTree(X[split['rightIdxs']], y[split['rightIdxs']], depth + 1)
        
        return self.Node(feature=split['feature'], threshold=split['threshold'], left=leftSubTree, right=rightSubTree)
        
    def fit(self, X, y):
        self.tree = self.buildTree(X, y)
    
    #traversing tree fo predictions
    def traverseTree(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self.traverseTree(x, node.left)
        return self.traverseTree(x, node.right)
    
    def predict(self, X):
        return np.array([self.traverseTree(x, self.tree) for x in X])