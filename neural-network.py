import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def initialize_parameters(input_size, hidden_size, output_size):
    W1 = np.random.randn(input_size, hidden_size) * 0.01
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, output_size) * 0.01
    b2 = np.zeros((1, output_size))
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

def forward_propagation(X, params):
    Z1 = np.dot(X, params["W1"]) + params["b1"]
    A1 = relu(Z1)
    Z2 = np.dot(A1, params["W2"]) + params["b2"]
    A2 = sigmoid(Z2)
    cache = (Z1, A1, Z2, A2)
    return A2, cache

def compute_cost(A2, Y):
    m = Y.shape[0]
    cost = -np.sum(Y * np.log(A2) + (1 - Y) * np.log(1 - A2)) / m
    return cost

def backward_propagation(X, Y, cache, params):
    Z1, A1, Z2, A2 = cache
    m = Y.shape[0]
    dZ2 = A2 - Y
    dW2 = np.dot(A1.T, dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m
    dZ1 = np.dot(dZ2, params["W2"].T) * relu_derivative(Z1)
    dW1 = np.dot(X.T, dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m
    gradients = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
    return gradients

def update_parameters(params, grads, learning_rate):
    params["W1"] -= learning_rate * grads["dW1"]
    params["b1"] -= learning_rate * grads["db1"]
    params["W2"] -= learning_rate * grads["dW2"]
    params["b2"] -= learning_rate * grads["db2"]
    return params

def model(X, Y, hidden_size, learning_rate, num_epochs):
    np.random.seed(2)
    input_size = X.shape[1]
    output_size = 1
    params = initialize_parameters(input_size, hidden_size, output_size)
    for i in range(num_epochs):
        A2, cache = forward_propagation(X, params)
        cost = compute_cost(A2, Y)
        grads = backward_propagation(X, Y, cache, params)
        params = update_parameters(params, grads, learning_rate)
        if i % 1000 == 0:
            print(f"Cost after iteration {i}: {cost:.4f}")
    return params

# Generate synthetic data
X, y = make_moons(n_samples=300, noise=0.20, random_state=0)
y = y.reshape(-1, 1)  # Reshape for consistency in matrix operations
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Hyperparameter tuning
learning_rates = [0.01, 0.001]
epochs = [10000, 15000]
best_params = None
lowest_cost = float('inf')

for lr in learning_rates:
    for epoch in epochs:
        print(f"Training with lr = {lr} and epochs = {epoch}")
        params = model(X_train, y_train, hidden_size=4, learning_rate=lr, num_epochs=epoch)
        _, cache = forward_propagation(X_val, params)
        val_cost = compute_cost(cache[-1], y_val)
        if val_cost < lowest_cost:
            best_params = params
            lowest_cost = val_cost

print("Best parameters found with lowest validation cost.")
