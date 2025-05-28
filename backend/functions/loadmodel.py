import os, csv
def load_model(output_dir):
    with open(os.path.join(output_dir,"mnist_model_weights.csv"), "r") as f:
        reader = csv.reader(f)
        W = [[float(val) for val in row] for row in reader]
    with open(os.path.join(output_dir,"mnist_model_biases.csv"), "r") as f:
        reader = csv.reader(f)
        b = [float(val) for val in next(reader)]
    return W, b