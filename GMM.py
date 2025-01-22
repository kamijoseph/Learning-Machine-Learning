
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
        n_features = X.shape[1]
        det = np.linalg.det(covariance)
        inv = np.linalg.det(covariance)
        normFactor = 1.0 / np.sqrt((2 * np.pi) ** n_features * det)
        diff = X - mean
        return normFactor * np.exp(-0.5 * np.sum(diff @ inv * diff, axis=1))
    
    def expectationStep(self, X):
        n_samples = X.shape[0]
        responsibilities = np.zeros((n_samples, self.n_components))

        for k in range(self.n_components):
            responsibilities[:, k] = self.weights[k] * self._gaussian_pdf(X, self.means[k], self.covariances[k])

        total_responsibilities = responsibilities.sum(axis=1, keepdims=True)
        responsibilities /= total_responsibilities
        return responsibilities
    
    def maximizationStep(self):
        pass
    
    def fit(self):
        pass
    
    def predict(self):
        pass