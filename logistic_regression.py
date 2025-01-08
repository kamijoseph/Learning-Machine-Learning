
# Simple Logistic regresion algorithm using synthetic data.
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generating the synthetic data and 500 samples with 2 features
np.random.seed(42)
n_samples = 500 

# Feature
X1 = np.random.normal(loc=2.5, scale=1.0, size=n_samples)
X2 = np.random.normal(loc=1.5, scale=1.5, size=n_samples)

# Decision boundary
y = (2.5 * X1 - 1.5 * X2 > 2.5).astype(int)

# Combining the features
X = np.column_stack((X1, X2))

# Upload and Save te data into a csv file
data = pd.DataFrame(data={"Feature1": X1, "Feature2": X2, "Target": y})
data.to_csv("synthetic_data1.csv", index=False)
print("Data saved to synthetic_data1.csv")

# The Logistic Regression algorithm
class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations = 1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        
    def sigmoid(self, z):
        return 1/(1 + np.exp(-z))
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.n_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)
            
            dw = (1/n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1/n_samples) * np.sum(y_predicted - y)
            
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)
        return [1 if i > 0.5 else 0 for i in y_predicted]
    
# Load the data and train the model
data = pd.read_csv('synthetic_data1.csv')
X = data[['Feature1', 'Feature2']].values
y = data['Target'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression(learning_rate=0.01, n_iterations=1000)
model.fit(X_train, y_train)

# Evaluating the model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy * 100:.2f}%")