import numpy as np

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components

    def fit_transform(self, X):
        # Compute covariance matrix
        cov_matrix = np.cov(X.T)
        # Compute eigenvectors and eigenvalues
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        # Sort eigenvectors by eigenvalues
        sorted_indices = np.argsort(eigenvalues)[::-1]
        sorted_eigenvectors = eigenvectors[:, sorted_indices]
        # Select top n_components eigenvectors
        self.components = sorted_eigenvectors[:, :self.n_components]
        # Project data onto principal components
        X_pca = np.dot(X, self.components)
        return X_pca

class TSNE:
    def __init__(self, n_components, perplexity=30, learning_rate=200, n_iter=1000):
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter

    def fit_transform(self, X):
        # Compute pairwise distances
        distances = self.compute_pairwise_distances(X)
        # Initialize low-dimensional embedding
        Y = np.random.randn(X.shape[0], self.n_components)
        # Perform gradient descent to minimize the KL divergence
        for i in range(self.n_iter):
            q_values = self.compute_q_values(Y)
            grad = self.compute_gradient(X, Y, distances, q_values)
            Y -= self.learning_rate * grad
        return Y

    def compute_pairwise_distances(self, X):
        n_samples = X.shape[0]
        distances = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                distances[i, j] = np.linalg.norm(X[i] - X[j])
        return distances

    def compute_q_values(self, Y):
        distances = self.compute_pairwise_distances(Y)
        inv_distances = 1 / (1 + distances**2)
        np.fill_diagonal(inv_distances, 0)
        return inv_distances / np.sum(inv_distances)

    def compute_gradient(self, X, Y, distances, q_values):
        grad = np.zeros_like(Y)
        n_samples = X.shape[0]
        for i in range(n_samples):
            grad_i = np.zeros_like(Y[i])
            for j in range(n_samples):
                grad_i += (q_values[i, j] - distances[i, j]) * (Y[i] - Y[j]) * (1 / (1 + distances[i, j]**2))
            grad[i] = 4 * np.sum(grad_i)
        return grad

# Example usage
if __name__ == "__main__":
    X = np.random.rand(100, 10)  # Example dataset with 100 samples and 10 features

    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    print("PCA transformed data shape:", X_pca.shape)

    # t-SNE
    tsne = TSNE(n_components=2)
    X_tsne = tsne.fit_transform(X)
    print("t-SNE transformed data shape:", X_tsne.shape)
