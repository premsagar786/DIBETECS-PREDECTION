import pandas as pd
from sklearn.model_selection import train_test_split
from models.knn import train_knn
from models.decision_tree import train_decision_tree
from models.mlp import train_mlp

# Load data
data = pd.read_csv('data/diabetes.csv')

# Split features and target
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
knn_model = train_knn(X_train, y_train, n_neighbors=9)
tree_model = train_decision_tree(X_train, y_train, max_depth=3)
mlp_model = train_mlp(X_train, y_train)

# Evaluate models
print(f"KNN Accuracy: {knn_model.score(X_test, y_test):.2f}")
print(f"Decision Tree Accuracy: {tree_model.score(X_test, y_test):.2f}")
print(f"MLP Accuracy: {mlp_model.score(X_test, y_test):.2f}")