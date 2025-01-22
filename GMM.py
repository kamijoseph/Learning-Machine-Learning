
# Gaussian Mixture Model Implementmentaion from scratch with numpy
import numpy as np

class GaussianMixtureModel:
    def __init__(self, n_components=1, maxIterations=100, tol=1e-6):
        self.n_components = n_components
        self.maxIterations = maxIterations
        self.tol = tol
        self.means = None
        self.covarinces = None
        self.weights = None
        
    
    def initializeParameters(self, X):
        n_samples, n_features = self.X.shape
        self.weights = np.ones(self.n_components) / self.n_components
        self.means = X[np.random.choice(n_samples, self.n_components, replace=False)]
        self.covarinces = np.array([np.eye(n_features) for _ in range(self.n_components)])
        
    
    def gaussianPdf(self, X, mean, covariance):
        pass
    
    def expectationStep(self):
        pass
    
    def maximizationStep(self):
        pass
    
    def fit(self):
        pass
    
    def predict(self):
        pass