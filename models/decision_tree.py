from sklearn.tree import DecisionTreeClassifier

def train_decision_tree(X_train, y_train, max_depth=3):
    tree = DecisionTreeClassifier(max_depth=max_depth)
    tree.fit(X_train, y_train)
    return tree