import numpy as np

def bootstrap_sample(X, y):
    n_samples = X.shape[0]
    indices = np.random.choice(n_samples, size=n_samples, replace=True)
    return X[indices], y[indices]

def most_common_label(y):
    (values, counts) = np.unique(y, return_counts=True)
    most_common = np.argmax(counts)
    return values[most_common]

class DecisionTreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None

def entropy(y):
    hist = np.bincount(y)
    ps = hist / np.sum(hist)
    return -np.sum([p * np.log2(p) for p in ps if p > 0])

def best_split(X, y, num_features):
    n_samples, n_features = X.shape
    best_gain = -1
    split_idx, split_thresh = None, None
    for feature_idx in np.random.choice(n_features, num_features, replace=False):
        thresholds = np.unique(X[:, feature_idx])
        for threshold in thresholds:
            left_idx = np.where(X[:, feature_idx] <= threshold)[0]
            right_idx = np.where(X[:, feature_idx] > threshold)[0]
            if len(left_idx) == 0 or len(right_idx) == 0:
                continue
            left_y, right_y = y[left_idx], y[right_idx]
            p_left, p_right = len(left_y) / n_samples, len(right_y) / n_samples
            curr_gain = entropy(y) - (p_left * entropy(left_y) + p_right * entropy(right_y))
            if curr_gain > best_gain:
                best_gain = curr_gain
                split_idx, split_thresh = feature_idx, threshold
    return split_idx, split_thresh

def build_tree(X, y, depth, max_depth, min_samples_split, num_features):
    if depth == max_depth or len(y) < min_samples_split or len(np.unique(y)) == 1:
        leaf_value = most_common_label(y)
        return DecisionTreeNode(value=leaf_value)
    
    feature_idx, threshold = best_split(X, y, num_features)
    if feature_idx is None:
        return DecisionTreeNode(value=most_common_label(y))
    
    left_idx = np.where(X[:, feature_idx] <= threshold)[0]
    right_idx = np.where(X[:, feature_idx] > threshold)[0]
    
    left = build_tree(X[left_idx, :], y[left_idx], depth + 1, max_depth, min_samples_split, num_features)
    right = build_tree(X[right_idx, :], y[right_idx], depth + 1, max_depth, min_samples_split, num_features)
    
    return DecisionTreeNode(feature_idx, threshold, left, right)

def predict(node, x):
    while not node.is_leaf_node():
        if x[node.feature] <= node.threshold:
            node = node.left
        else:
            node = node.right
    return node.value

class RandomForestClassifier:
    def __init__(self, n_trees=100, max_depth=10, min_samples_split=2, num_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.num_features = num_features
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            X_sample, y_sample = bootstrap_sample(X, y)
            if self.num_features is None:
                self.num_features = int(np.sqrt(X.shape[1]))
            tree = build_tree(X_sample, y_sample, 0, self.max_depth, self.min_samples_split, self.num_features)
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = np.array([[predict(tree, x) for tree in self.trees] for x in X])
        return np.array([most_common_label(tree_pred) for tree_pred in tree_preds])

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load the Iris dataset
data = load_iris()
X, y = data.data, data.target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the RandomForestClassifier
clf = RandomForestClassifier(n_trees=10, max_depth=10)
clf.fit(X_train, y_train)

# Predict and calculate accuracy
y_pred = clf.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")
