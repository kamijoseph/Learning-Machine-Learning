
# Naive Bayes Algorithm with Numpy from Scratch.
import numpy as np
from collections import defaultdict

class NaiveBayes:
    def __init__(self):
        self.classPriors = {}
        self.featureLikelihoods = defaultdict(lambda: defaultdict(float))
        self.classes = []
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        
        # calculate the class prior and the feature likelihood
        for cls in self.classes:
            self.classPriors[cls] = np.sum(y == cls) / n_samples
            
        # feature likelihood per class
        for cls in self.classes:
            X_cls = X[y == cls]
            for featureIdx in range(n_features):
                featureValues = X_cls[:, featureIdx]
                mean = np.mean(featureValues)
                variance = np.var(featureValues)
                self.featureLikelihoods[cls][featureIdx] = (mean, variance)
                
    
    def calculateLikelihood(self):
        pass
    
    def calculatePosterior(self):
        pass
    
    def predict(self):
        pass