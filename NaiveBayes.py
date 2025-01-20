
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
                
    
    def calculateLikelihood(self, x, mean,variance):
        # calculating using gaussian probability density function
        eps = 1e-9
        coefficient = 1.0 / np.sqrt(2.0 * np.pi * (variance + eps))
        exponent = np.exp(-((x - mean) ** 2) / (2.0 * (variance + eps)))
        return coefficient * exponent
        
    
    def calculatePosterior(self, x):
        # Poterior probability for each class given a sample x
        posteriors = {}
        for cls in self.classes:
            prior = np.log(self.classPriors[cls])
            likelihood = 0
            for featureIdx, featureValue in enumerate(x):
                mean, variance = self.featureLikelihoods[cls][featureIdx]
                likelihood +=np.log((self.calculateLikelihood(featureValue, mean, variance)))
            posteriors[cls] = prior + likelihood
        return posteriors
    
    def predict(self, X):
        predictions = []
        for x in X:
            posteriors = self.calculatePosterior(x)
            predictions.append(max(posteriors, key=posteriors.get))
        return np.array(predictions)