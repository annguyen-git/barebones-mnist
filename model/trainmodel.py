import os
import numpy as np
import pandas as pd
import csv

DATA_DIR = os.path.join(os.path.dirname(__file__), "mnist")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "backend/parameters")

def load_mnist_csv(path):
    df = pd.read_csv(path)
    labels = df.iloc[:, 0].values
    images = df.iloc[:, 1:].values
    return images, labels

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def one_hot_encode(labels, num_classes=10):
    return np.eye(num_classes)[labels]

def train(X, y, epochs=50, lr=0.1, batch_size=64):
    num_samples, num_features = X.shape
    num_classes = 10
    # Initialize weights randomly
    W = np.random.randn(num_features, num_classes) * 0.01
    b = np.zeros(num_classes)
    y_one_hot = one_hot_encode(y, num_classes)

    for epoch in range(epochs):
        # Shuffle data
        perm = np.random.permutation(num_samples)
        X_shuffled = X[perm]
        y_shuffled = y_one_hot[perm]

        for i in range(0, num_samples, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]

            scores = np.dot(X_batch, W) + b
            probs = softmax(scores)

            error = probs - y_batch
            dW = np.dot(X_batch.T, error) / X_batch.shape[0]
            db = np.mean(error, axis=0)

            W -= lr * dW
            b -= lr * db

        if (epoch + 1) % 10 == 0:
            # Compute loss on full data for logging
            scores_full = np.dot(X, W) + b
            probs_full = softmax(scores_full)
            loss = -np.sum(y_one_hot * np.log(probs_full + 1e-9)) / num_samples
            print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

    return W, b

def predict(X, W, b):
    scores = np.dot(X, W) + b
    return np.argmax(scores, axis=1)

def save_model_no_numpy(W, b):
    with open(os.path.join(OUTPUT_DIR,"mnist_model_weights.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(W)
    with open(os.path.join(OUTPUT_DIR,"mnist_model_biases.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(b)

train_images, train_labels = load_mnist_csv(os.path.join(DATA_DIR, "mnist_train.csv"))
test_images, test_labels = load_mnist_csv(os.path.join(DATA_DIR, "mnist_test.csv"))

threshold = 51
X_train = (train_images > threshold).astype(int)
y_train = train_labels.astype(np.int64)

X_test = (test_images > threshold).astype(int)
y_test = test_labels.astype(np.int64)

print("Start training...")
W, b = train(X_train, y_train, epochs=100, lr=0.1)

print("Evaluating on test set...")
y_pred = predict(X_test, W, b)
accuracy = np.mean(y_pred == y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")


save_model_no_numpy(W, b)
