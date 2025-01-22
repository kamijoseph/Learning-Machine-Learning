
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
        n_samples, n_features = X.shape
        self.weights = np.ones(self.n_components) / self.n_components
        self.means = X[np.random.choice(n_samples, self.n_components, replace=False)]
        self.covariances = np.array([np.eye(n_features) for _ in range(self.n_components)])
    
    def gaussianPdf(self, X, mean, covariance):
        n_features = X.shape[1]
        det = np.linalg.det(covariance)
        inv = np.linalg.inv(covariance)
        normFactor = 1.0 / np.sqrt((2 * np.pi) ** n_features * det)
        diff = X - mean
        return normFactor * np.exp(-0.5 * np.sum(diff @ inv * diff, axis=1))
    
    def expectationStep(self, X):
        n_samples = X.shape[0]
        responsibilities = np.zeros((n_samples, self.n_components))
        for k in range(self.n_components):
            responsibilities[:, k] = self.weights[k] * self.gaussianPdf(X, self.means[k], self.covariances[k])
        totalResponsibilities = responsibilities.sum(axis=1, keepdims=True) + 1e-10
        responsibilities /= totalResponsibilities
        return responsibilities
    
    def maximizationStep(self, X, responsibilities):
        n_samples = X.shape[0]
        for k in range(self.n_components):
            resp_k = responsibilities[:, k]
            total_resp_k = resp_k.sum()
            
            self.weights[k] = total_resp_k / n_samples
            self.means[k] = np.sum(resp_k[:, np.newaxis] * X, axis=0) / total_resp_k

            diff = X - self.means[k]
            self.covariances[k] = np.dot((resp_k[:, np.newaxis] * diff).T, diff) / total_resp_k
    
    def fit(self, X):
        self.initializeParameters(X)

        for iteration in range(self.maxIterations):
            responsibilities = self.expectationStep(X)

            prevMeans = self.means.copy()
            self.maximizationStep(X, responsibilities)

            meanShift = np.linalg.norm(self.means - prevMeans)
            if meanShift < self.tol:
                break
    
    def predict(self, X):
        responsibilities = self.expectationStep(X)
        return np.argmax(responsibilities, axis=1)
    
np.random.seed(42)

#generate synthetic 2D data
n_samples = 300
X1 = np.random.multivariate_normal(mean=[0, 0], cov=[[1, 0], [0, 1]], size=n_samples // 3)
X2 = np.random.multivariate_normal(mean=[5, 5], cov=[[1, 0.5], [0.5, 1]], size=n_samples // 3)
X3 = np.random.multivariate_normal(mean=[-5, 5], cov=[[1, -0.5], [-0.5, 1]], size=n_samples // 3)
X = np.vstack([X1, X2, X3])

#fitting GMM
gmm = GaussianMixtureModel(n_components=3)
gmm.fit(X)

labels = gmm.predict(X)

#visualizing
import matplotlib.pyplot as plt
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=30, alpha=0.7)
plt.scatter(gmm.means[:, 0], gmm.means[:, 1], color='red', marker='x', s=100, label='Cluster Centers')
plt.title("Gaussian Mixture Model Clustering")
plt.legend()
plt.show()