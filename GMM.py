
# Gaussian Mixture Model Implementmentaion from scratch with numpy
class GaussianMixtureModel:
    def __init__(self, n_components=1, maxIterations=100, tol=1e-6):
        self.n_components = n_components
        self.maxIterations = maxIterations
        self.tol = tol
        self.means = None
        self.covarinces = None
        self.weights = None
        
    
    def initializeParameters(self):
        pass
    
    def gaussianPdf(self):
        pass
    
    def expectationStep(self):
        pass
    
    def maximizationStep(self):
        pass
    
    def fit(self):
        pass
    
    def predict(self):
        pass