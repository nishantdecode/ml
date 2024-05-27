import numpy as np

def sigmoid(z):
    """The Sigmoid function."""
    return 1 / (1 + np.exp(-z))

def predict(features, weights):
    """Returns 1D array of probabilities that the class label == 1."""
    z = np.dot(features, weights)
    return sigmoid(z)

def cost_function(features, labels, weights):
    """Compute the cost over the whole dataset."""
    observations = len(labels)
    predictions = predict(features, weights)
    # Class 1 cost
    class1_cost = -labels * np.log(predictions)
    # Class 0 cost
    class0_cost = -(1 - labels) * np.log(1 - predictions)
    # Cost is the average of class1_cost and class0_cost
    cost = (class1_cost + class0_cost).sum() / observations
    return cost

def update_weights(features, labels, weights, lr):
    """Vectorized Gradient Descent."""
    N = len(features)
    # Get Predictions
    predictions = predict(features, weights)
    # Gradient calculation
    gradient = np.dot(features.T, predictions - labels) / N
    # Update weights
    weights -= lr * gradient
    return weights

def train(features, labels, weights, lr, iters):
    """Training function."""
    for i in range(iters):
        weights = update_weights(features, labels, weights, lr)
        # Print Log-likelihood every so often
        if i % 100 == 0:
            cost = cost_function(features, labels, weights)
            print(f"Iteration {i}: Cost {cost}")
    return weights

# Dummy data
features = np.array([
    [1, 2],
    [1, 3],
    [2, 3],
    [6, 5],
    [7, 8],
    [9, 8],
])

labels = np.array([0, 0, 0, 1, 1, 1])

# Hyperparameters
lr = 0.1
iters = 1000
weights = np.zeros(features.shape[1])

# Train the model
weights = train(features, labels, weights, lr, iters)

# Print out the final weights
print(f"Trained weights: {weights}")
