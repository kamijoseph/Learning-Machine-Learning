
# Simple Support Vector Macine Classifier for 2-Dimensional Data from Scratch.
import numpy as np
import matplotlib.pyplot as plt

class SupportVectorMachine:
    def __init__(self, learningRate=0.001, lambdaParameter=0.01, n_iterations=1000):
        self.learningRate = learningRate
        self.lambdaParameter = lambdaParameter
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # labels --> +1 and -1
        y_ = np.where(y <= 0, -1, 1)
        
        for _ in range(self.n_iterations):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.weights) - self.bias) >= 1
                
                # point correctly classified ?
                if condition:
                    self.weights = self.learningRate * (2 * self.lambdaParameter * self.weights)
                # incorrectly classified ?
                else:
                    self.weights -= self.learningRate * (2 * self.lambdaParameter * self.weights - np.dot(x_i, y_[idx]))
    
    def predict(self, X):
        approximate = np.dot(X, self.weights) - self.bias
        return np.sign(approximate)
    
# Synthdata.....................
np.random.seed(42)
X = np.random.randn(100, 2)
y = np.array([1 if x[0] * 2 + x[1] > 1 else 0 for x in X]) 

# Visualizing....................
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm')
plt.title("Synthetic Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show

# Train dragon, sorry Train Model
svm = SupportVectorMachine(learningRate=.001, lambdaParameter=0.01, n_iterations=1000)
svm.fit(X, y)

# Visualizing desicion bound................
def plotDesBound(X, y, model):
    x0 = np.linspace(min(X[:, 0]), max(X[:, 0]), 100)
    x1 = -(model.weights[0] * x0 + model.bias) / model.weights[1]
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm')
    plt.plot(x0, x1, '-k', label="Decision Boundary")
    plt.title("Support Vector Machine Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.show()
plotDesBound()

# Evaluating the predictions.................
predictions = svm.predict(X)
accuracy = np.mean(predictions == np.where(y == 0, -1, 1))
print(f"Accuracy: {accuracy * 100:.2f}%")
