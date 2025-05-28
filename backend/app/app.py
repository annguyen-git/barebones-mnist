from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from functions.loadmodel import load_model
from functions.predictmodel import predict_model
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "parameters")

W, b = load_model(OUTPUT_DIR)

app = Flask(__name__, static_folder=os.path.abspath(os.path.join(os.path.dirname(__file__), '../../frontend')))
CORS(app, resources={r"/predict": {"origins": "*"}})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"error": "Missing 'image' data"}), 400

    x = data["image"]
    if len(x) != 784:
        return jsonify({"error": "Input must be 28x28 flattened image (784 values)."}), 400

    pred = predict_model([x], W, b)
    return jsonify({"prediction": pred[0]})

@app.route('/')
def serve_index():
    # Serve your frontend's interface.html as main page
    return send_from_directory(app.static_folder, 'interface.html')

@app.route('/<path:path>')
def serve_static(path):
    # Serve all other static files from frontend folder
    return send_from_directory(app.static_folder, path)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
