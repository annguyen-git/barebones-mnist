import csv
import os
import math

DATA_DIR = os.path.expanduser("mnist")

def load_mnist_csv_no_lib(path):
    images = []
    labels = []
    with open(path, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            labels.append(int(row[0]))
            images.append([int(pix) for pix in row[1:]])
    return images, labels

def softmax_no_lib(z):
    max_z = max(z)
    exp_z = [math.exp(i - max_z) for i in z]
    sum_exp_z = sum(exp_z)
    return [i / sum_exp_z for i in exp_z]

def one_hot_encode_no_lib(labels, num_classes=10):
    one_hot = []
    for label in labels:
        vec = [0] * num_classes
        vec[label] = 1
        one_hot.append(vec)
    return one_hot

def dot_product(vec, mat):
    # vec: length n, mat: n x m -> returns vector length m
    result = []
    for col in range(len(mat[0])):
        s = 0
        for i in range(len(vec)):
            s += vec[i] * mat[i][col]
        result.append(s)
    return result

def matrix_transpose(mat):
    return [[mat[row][col] for row in range(len(mat))] for col in range(len(mat[0]))]

def matrix_multiply(matA, matB):
    # matA: a x b, matB: b x c -> returns a x c
    result = []
    for i in range(len(matA)):
        row = []
        for j in range(len(matB[0])):
            s = 0
            for k in range(len(matB)):
                s += matA[i][k] * matB[k][j]
            row.append(s)
        result.append(row)
    return result

def vector_subtract(v1, v2):
    return [a - b for a, b in zip(v1, v2)]

def vector_add(v1, v2):
    return [a + b for a, b in zip(v1, v2)]

def scalar_multiply_vector(scalar, vec):
    return [scalar * x for x in vec]

def scalar_multiply_matrix(scalar, mat):
    return [[scalar * x for x in row] for row in mat]

def mean_vector(vectors):
    length = len(vectors)
    summed = [0] * len(vectors[0])
    for v in vectors:
        for i in range(len(v)):
            summed[i] += v[i]
    return [x / length for x in summed]

def train_no_lib(X, y, epochs=50, lr=0.1, batch_size=64):
    num_samples = len(X)
    num_features = len(X[0])
    num_classes = 10
    

    import random
    W = [[random.gauss(0, 0.01) for _ in range(num_classes)] for _ in range(num_features)]
    b = [0] * num_classes
    
    y_one_hot = one_hot_encode_no_lib(y, num_classes)
    
    for epoch in range(epochs):

        combined = list(zip(X, y_one_hot))
        random.shuffle(combined)
        X_shuffled, y_shuffled = zip(*combined)
        
        for i in range(0, num_samples, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            
     
            scores_batch = [vector_add(dot_product(x, W), b) for x in X_batch]
            probs_batch = [softmax_no_lib(s) for s in scores_batch]
            

            dW = [[0] * num_classes for _ in range(num_features)]
            db = [0] * num_classes
            
            batch_len = len(X_batch)
            
            for x, prob, y_true in zip(X_batch, probs_batch, y_batch):
                error = vector_subtract(prob, y_true) 
                for f in range(num_features):
                    for c in range(num_classes):
                        dW[f][c] += x[f] * error[c]
                for c in range(num_classes):
                    db[c] += error[c]
            
        
            dW = scalar_multiply_matrix(1.0 / batch_len, dW)
            db = [val / batch_len for val in db]
            
    
            for f in range(num_features):
                for c in range(num_classes):
                    W[f][c] -= lr * dW[f][c]
            b = [b_i - lr * db_i for b_i, db_i in zip(b, db)]
        
        if (epoch + 1) % 10 == 0:
            scores_full = [vector_add(dot_product(x, W), b) for x in X]
            probs_full = [softmax_no_lib(s) for s in scores_full]
            loss = 0
            for y_vec, p_vec in zip(y_one_hot, probs_full):
                for yt, p in zip(y_vec, p_vec):
                    loss -= yt * math.log(p + 1e-9)
            loss /= num_samples
            print(f"Epoch {epoch+1}, Loss: {loss:.4f}")
    
    return W, b

def predict_no_lib(X, W, b):
    predictions = []
    for x in X:
        scores = vector_add(dot_product(x, W), b)
        pred = scores.index(max(scores))
        predictions.append(pred)
    return predictions

train_images, train_labels = load_mnist_csv_no_lib(os.path.join(DATA_DIR, "mnist_train.csv"))
test_images, test_labels = load_mnist_csv_no_lib(os.path.join(DATA_DIR, "mnist_test.csv"))

X_train = [[pix / 255.0 for pix in img] for img in train_images]
y_train = train_labels

X_test = [[pix / 255.0 for pix in img] for img in test_images]
y_test = test_labels

print("Start training...")
W, b = train_no_lib(X_train, y_train, epochs=100, lr=0.1)

print("Evaluating on test set...")
y_pred = predict_no_lib(X_test, W, b)
accuracy = sum([pred == true for pred, true in zip(y_pred, y_test)]) / len(y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
