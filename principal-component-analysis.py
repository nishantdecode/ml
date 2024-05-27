import numpy as np

def pca(X, n_components=2):
    """
    Perform Principal Component Analysis (PCA) to reduce the
    dimensionality of the input data.
    """
    # Center the data
    X_mean = np.mean(X, axis=0)
    X_centered = X - X_mean
    # Compute covariance matrix
    covariance_matrix = np.cov(X_centered, rowvar=False)
    # Perform eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    # Sort eigenvalues and eigenvectors in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    # Choose the top n_components eigenvectors
    principal_components = eigenvectors[:, :n_components]
    # Project the data onto the principal components
    X_pca = np.dot(X_centered, principal_components)
    return X_pca

def compute_pairwise_distances(X):
    """
    Compute pairwise Euclidean distances between all pairs of points in X.
    """
    n = X.shape[0]
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            distances[i, j] = np.linalg.norm(X[i] - X[j])
    return distances

def compute_probabilities(distances, perplexity, epsilon=1e-8):
    """
    Compute conditional probabilities Pij given the pairwise distances
    and perplexity.
    """
    n = distances.shape[0]
    probabilities = np.zeros((n, n))
    for i in range(n):
        distances[i, i] = 0
        # Compute conditional probabilities for point i
        numerator = np.exp(-distances[i] / (2 * perplexity))
        denominator = np.sum(numerator) - 1  # sum except i itself
        probabilities[i] = numerator / (denominator + epsilon)
    return probabilities

def tSNE(X, n_components=2, perplexity=30, learning_rate=100,
         n_iters=1000, verbose=True, clip_gradients=True, lr_schedule=False,
         early_stopping=False):
    """
    Perform t-SNE to reduce the dimensionality of the input data.
    """
    # Initialize low-dimensional embeddings randomly
    np.random.seed(0)
    Y = np.random.randn(X.shape[0], n_components)
    # Compute pairwise Euclidean distances
    distances = compute_pairwise_distances(X)
    # Initialize previous iteration's gains
    gains = np.ones_like(Y)
    prev_cost = np.inf
    for i in range(n_iters):
        # Compute pairwise probabilities
        probabilities = compute_probabilities(distances, perplexity)
        # Compute joint probabilities
        Q = (1 / (1 + np.sum(np.square(Y[:, np.newaxis, :] -
             Y[np.newaxis, :, :]), axis=-1))) / 2
        np.fill_diagonal(Q, 0)
        # Compute gradients
        PQ_diff = probabilities - Q
        grad = np.zeros_like(Y)
        for j in range(X.shape[0]):
            grad[j] = np.sum((PQ_diff[j, :, np.newaxis] * (Y[j] - Y)).T, axis=1)
        # Gradient clipping
        if clip_gradients:
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 1.0:
                grad /= grad_norm
        # Update the learning rate
        if lr_schedule:
            learning_rate *= 0.9
        # Update the low-dimensional embeddings
        gains = (gains + 0.2) * ((grad > 0) != (gains > 0)) + (gains *
                0.8) * ((grad > 0) == (gains > 0))
        Y -= learning_rate * gains * grad
        # Compute cost
        cost = np.sum(probabilities * np.log((probabilities + 1e-8) /
                     (Q + 1e-8)))
        # Print progress
        if verbose and i % 100 == 0:
            print(f"Iteration {i}: Cost = {cost}")
        # Early stopping
        if early_stopping and np.abs(cost - prev_cost) < 1e-4:
            print("Early stopping at iteration", i)
            break
        prev_cost = cost
    return Y

# Example usage:
if __name__ == "__main__":
    # Generate some random data
    np.random.seed(0)
    X = np.random.randn(100, 10)
    # Reduce dimensions using PCA
    X_pca = pca(X, n_components=2)
    print("PCA results:")
    print(X_pca)
    # Reduce dimensions using t-SNE
    X_tsne = tSNE(X, n_components=2)
    print("\nt-SNE results:")
    print(X_tsne)
