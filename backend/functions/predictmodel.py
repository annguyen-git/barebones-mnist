def dot_product(vec, mat):
    result = []
    for col in range(len(mat[0])):
        s = 0
        for i in range(len(vec)):
            s += vec[i] * mat[i][col]
        result.append(s)
    return result

def vector_add(v1, v2):
    return [a + b for a, b in zip(v1, v2)]

def predict_model(X, W, b):
    predictions = []
    for x in X:
        scores = vector_add(dot_product(x, W), b)
        pred = scores.index(max(scores))
        predictions.append(pred)
    return predictions