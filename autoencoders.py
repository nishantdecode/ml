import numpy as np

class Autoencoder:
    def __init__(self, input_dim, encoding_dim, learning_rate=0.01, epochs=50):
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        # Initialize weights and biases
        self.W1 = np.random.randn(encoding_dim, input_dim)
        self.b1 = np.zeros((encoding_dim, 1))
        self.W2 = np.random.randn(input_dim, encoding_dim)
        self.b2 = np.zeros((input_dim, 1))
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def train(self, X):
        for epoch in range(self.epochs):
            # Forward pass
            encoded_output = self.sigmoid(np.dot(self.W1, X.T) + self.b1)
            decoded_output = self.sigmoid(np.dot(self.W2, encoded_output) + self.b2)
            
            # Backpropagation
            error = X.T - decoded_output
            delta2 = error * self.sigmoid_derivative(decoded_output)
            delta1 = np.dot(self.W2.T, delta2) * self.sigmoid_derivative(encoded_output)
            
            # Update weights and biases
            self.W2 += self.learning_rate * np.dot(delta2, encoded_output.T)
            self.b2 += self.learning_rate * np.sum(delta2, axis=1, keepdims=True)
            self.W1 += self.learning_rate * np.dot(delta1, X)
            self.b1 += self.learning_rate * np.sum(delta1, axis=1, keepdims=True)
            
            # Compute and print the mean squared error
            mse = np.mean(np.square(error))
            print(f"Epoch {epoch+1}/{self.epochs}, Mean Squared Error: {mse}")
    
    def compress(self, X):
        # Forward pass to get the encoded representation
        return self.sigmoid(np.dot(self.W1, X.T) + self.b1).T
    
    def decompress(self, encoded_data):
        # Forward pass to reconstruct the original data
        return self.sigmoid(np.dot(self.W2, encoded_data.T) + self.b2).T

if __name__ == "__main__":
    # Generate random data
    np.random.seed(0)
    data = np.random.rand(1000, 10)  # Example dataset with 1000 samples and 10 features
    
    # Initialize and train the autoencoder
    autoencoder = Autoencoder(input_dim=10, encoding_dim=5, learning_rate=0.01, epochs=50)
    autoencoder.train(data)
    
    # Compress the data
    compressed_data = autoencoder.compress(data)
    
    # Decompress the data
    decompressed_data = autoencoder.decompress(compressed_data)
    
    # Print original and reconstructed data for comparison
    print("\nOriginal Data:")
    print(data)
    print("\nReconstructed Data:")
    print(decompressed_data)
