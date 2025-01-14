
# Implementing Principal Component Analysis
import numpy as np
import matplotlib.pyplot as plt

#synthetic data, and features init
np.random.seed(42)
mean = [0, 0]
covariance = [[1, 0.8], [0.8, 1]]
X = np.random.multivariate_normal(mean, covariance, 100)

#Standardizing data....................
def standardizeData(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return (data - mean) / std
x_standardized = standardizeData(X)

#Covariance Matrix
def compCovMatrix(data):
    n_samples = data.shape[0]
    covarianceMatrix = (1 / (n_samples - 1)) * np.dot(data.T, data)
    return covarianceMatrix
covarianceMatrix = compCovMatrix(x_standardized)

#Eigen vectors and Values go brrrr!
def eigenComp(matrix):
    eigenValues, eigenVectors = np.linalg.eig(matrix)
    return eigenValues, eigenVectors
eigenValues, eigenVectors = eigenComp(covarianceMatrix)

#sorting Eigenvalues, selecting top Components
def topComponent(eigenValues, eigenVectors, numComponents):
    sortedInd = np.argsort(eigenValues)[::-1]
    topEigenValues = eigenValues[sortedInd[:numComponents]]
    topEigenVectors = eigenVectors[:, sortedInd[:numComponents]]
    return topEigenValues, topEigenVectors
topEigenValues, topEigenVectors = topComponent(eigenValues, eigenVectors, numComponents=1)

#mapping the data
def mapData(data, eigenVectors):
    return np.dot(data, eigenVectors)

X_pca = mapData(x_standardized, topEigenVectors)

# visualizing original and mapped data
plt.figure(figsize=(10, 5))