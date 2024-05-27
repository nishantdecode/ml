import numpy as np

class KMeans:
    def __init__(self, n_clusters, max_iter=300):
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def fit(self, X):
        n_samples, n_features = X.shape
        # Initialize cluster centroids randomly
        self.centroids = X[np.random.choice(n_samples, self.n_clusters, replace=False)]
        for _ in range(self.max_iter):
            # Assign each sample to the nearest centroid
            labels = self._assign_clusters(X)
            # Update cluster centroids
            new_centroids = np.array([X[labels == k].mean(axis=0) for k in range(self.n_clusters)])
            # Check for convergence
            if np.allclose(self.centroids, new_centroids):
                break
            self.centroids = new_centroids
        return labels

    def _assign_clusters(self, X):
        distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distances, axis=0)

class HierarchicalClustering:
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters

    def fit(self, X):
        n_samples = X.shape[0]
        # Initialize each sample as a cluster
        clusters = [[i] for i in range(n_samples)]
        while len(clusters) > self.n_clusters:
            # Compute pairwise distances between clusters
            distances = np.zeros((len(clusters), len(clusters)))
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    distances[i, j] = distances[j, i] = self._distance(X[clusters[i]], X[clusters[j]])
            # Find the closest clusters to merge
            min_dist = np.inf
            closest_clusters = None
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    if distances[i, j] < min_dist:
                        min_dist = distances[i, j]
                        closest_clusters = (i, j)
            # Merge the closest clusters
            clusters[closest_clusters[0]].extend(clusters[closest_clusters[1]])
            del clusters[closest_clusters[1]]
        # Assign labels to samples
        labels = np.zeros(n_samples)
        for i, cluster in enumerate(clusters):
            labels[cluster] = i
        return labels.astype(int)

    def _distance(self, cluster1, cluster2):
        return np.linalg.norm(np.mean(cluster1, axis=0) - np.mean(cluster2, axis=0))

# Example usage
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(0)
    X = np.random.randn(100, 2)
    
    # K-means clustering
    kmeans = KMeans(n_clusters=3)
    kmeans_labels = kmeans.fit(X)
    print("K-means Cluster labels:")
    print(kmeans_labels)
    
    # Hierarchical clustering
    hierarchical = HierarchicalClustering(n_clusters=3)
    hierarchical_labels = hierarchical.fit(X)
    print("\nHierarchical Cluster labels:")
    print(hierarchical_labels)
