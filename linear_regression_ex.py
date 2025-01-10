import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from LinearRegression import LinearRegression

#Generate synthetic regression dataset
X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

#Visualization of the data
plt.figure(figsize=(8.6))
plt.scatter(X[:, 0], y, color="b", marker="o", s=30)
plt.title("Scatter Plot of Dataset")
plt.xlabel("Feature")
plt.ylabel("Target")
plt.show()

#Create and train the model
reg = LinearRegression()
reg.fit(X_train, y_train)

#Predict on the test set
predictions = reg.predict(X_test)

#Mean Squared Error (MSE)
def calculate_mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

mse_value = calculate_mse(y_test, predictions)
print(f"The Mean Squared Error is: {mse_value}")