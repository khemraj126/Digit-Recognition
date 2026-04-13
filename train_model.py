from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
import pickle

print(" Loading MNIST...")

mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X = mnist.data / 255.0
y = mnist.target.astype(int)

# Use more data
X_train = X[:20000]
y_train = y[:20000]

X_test = X[20000:22000]
y_test = y[20000:22000]

print(" Training MLP...")

model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=20)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(" Accuracy:", accuracy)

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print(" Model saved!")