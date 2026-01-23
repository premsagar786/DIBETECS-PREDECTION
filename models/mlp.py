from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

def train_mlp(X_train, y_train):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    mlp = MLPClassifier()
    mlp.fit(X_train_scaled, y_train)
    return mlp