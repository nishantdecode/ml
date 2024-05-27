import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class GradientBoostingClassifier:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def loss_gradient(self, y, y_pred):
        return y - self.sigmoid(y_pred)

    def fit(self, X, y):
        # Initialize with the average of the target values
        init_pred = np.mean(y)
        residuals = y - init_pred
        for _ in range(self.n_estimators):
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)
            # Update predictions with the learning rate
            tree_pred = self.learning_rate * tree.predict(X)
            residuals -= tree_pred
            self.trees.append(tree)

    def predict_proba(self, X):
        y_pred = np.sum([tree.predict(X) for tree in self.trees], axis=0)
        return self.sigmoid(y_pred)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

# Example usage
# Load data
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train gradient boosting classifier
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
