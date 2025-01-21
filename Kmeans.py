
# K Means Clustering Algorithm
import numpy as np
import matplotlib.pyplot as plt

class KMeans:
    def __init__(self, n_clusters=3, maxIterations=100, tolerance=1e-4):
        self.n_clusters = n_clusters
        self.maxIterations = maxIterations
        self.tolerance = tolerance
        self.centroids = None
    
    def initializeCentroids(self, X):
        indices = np.random.choice(self.X.shape[0], self.n_clusters, replace=False)
        return X[indices]
    
    def calculateDistance(self, X, centroids):
        distances = np.sqrt(((X[:, np.newaxis, :] - centroids) ** 2).sum(axis=2))
        return distances
    
    def fit(self, X):
        self.centroids = self.initializeCentroids(X)
        for iters in range(self.maxIterations):
            # assign points to the nearest centroid
            distances = self.calculateDistance(X, self.centroids)
            clusters = np.argmin(distances, axis=1)
            # update centroids
            newCentroids = np.array([X[clusters == k].mean(axis=0) if np.any(clusters == k) else self.centroids[k]
                                     for k in range(self.n_clusters)])
            if np.all(np.abs(newCentroids - self.centroids) < self.tolerance):
                break
            self.centroids = newCentroids
        self.clusters = clusters
        
    def predict(self, X):
        distances = self.calculateDistance(X, self.centroids)
        return np.argmin(distances, axis=1)

# Step 2: Generate Synthetic Data
np.random.seed(42)

# Create synthetic data with 3 clusters
cluster1 = np.random.normal(loc=[2, 2], scale=0.5, size=(50, 2))
cluster2 = np.random.normal(loc=[8, 8], scale=0.5, size=(50, 2))
cluster3 = np.random.normal(loc=[5, 12], scale=0.5, size=(50, 2))

X = np.vstack((cluster1, cluster2, cluster3))

# Step 3: Apply K-Means
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)

# Step 4: Visualize the Results
plt.figure(figsize=(8, 6))
colors = ['r', 'g', 'b']
for k in range(3):
    cluster_points = X[kmeans.cluster_assignments == k]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=colors[k], label=f'Cluster {k+1}')

plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], c='black', s=200, marker='X', label='Centroids')
plt.title("K-Means Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()
