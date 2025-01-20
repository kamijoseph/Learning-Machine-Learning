
# Simple K-Nearest Neighbours algorithm
import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k=3):
        self.k = k
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def predict(self, X):
        predictions = [self.predictor(x) for x in X]
        return np.array(predictions)
    
    def predictor(self, x):
        distance = [np.linalg.norm(x-x_train) for x_train in self.X_train]
        k_indices = np.argsort(distance)[:self.k]
        k_nearestLabels = [self.y_train[i] for i in k_indices]
        mostCommon = Counter(k_nearestLabels).most_common(1)
        return mostCommon[0][0]

if __name__ == "__main__":
    np.random.seed(42)
    X_train = np.random.rand(10, 2) * 10  # 10 training points with 2 features each
    y_train = np.random.choice([0, 1], size=10)  # Binary target labels

    X_test = np.random.rand(5, 2) * 10  # 5 test points with 2 features each

    # train model
    knn = KNN(k=3)
    knn.fit(X_train, y_train)

    # Prediciti
    predictions = knn.predict(X_test)

    print("training data (features):\n", X_train)
    print("training labels:\n", y_train)
    print("test data (features):\n", X_test)
    print("predicted labels:\n", predictions)