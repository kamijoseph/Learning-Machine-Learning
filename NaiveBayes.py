
# Naive Bayes Algorithm with Numpy from Scratch.
import numpy as np
from collections import defaultdict

class NaiveBayes:
    def __init__(self):
        self.classPriors = {}
        self.featureLikelihoods = defaultdict(lambda: defaultdict(float))
        self.classes = []
    
    def fit(self):
        pass
    
    def calculateLikelihood(self):
        pass
    
    def calculatePosterior(self):
        pass
    
    def predict(self):
        pass