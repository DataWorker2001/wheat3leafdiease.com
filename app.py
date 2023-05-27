## -*- coding: utf-8 -*-
"""
Author: Arpan DevBro

Created on 27th May, 2023
"""

from __future__ import division, print_function
import sys
import os
from pathlib import Path
import numpy as np

# Keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Define a flask app
app = Flask(__name__)
CORS(app)

# Model saved with Keras model.save()
MODEL_PATH = 'model1.h5'

# Load your trained model
model = load_model(MODEL_PATH)

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file found'})

    f = request.files['file']

    if f.filename == '':
        return jsonify({'error': 'No file selected'})

    basepath = Path(os.getcwd())
    file_path = basepath / 'uploads' / secure_filename(f.filename)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    f.save(str(file_path))

    preds = model_predict(str(file_path), model)

    pred_class = np.argmax(preds)
    if pred_class == 0:
        res = ['Brown Rust infected Wheat Leaf with confidence of ' + str(float(np.max(preds) * 100))[:6] + "%"]
    elif pred_class == 1:
        res = ['Wheat Healthy Leaf with confidence of ' + str(float(np.max(preds) * 100))[:6] + "%"]
    elif pred_class == 2:
        res = ['Yellow Rust infected Wheat Leaf with confidence of ' + str(float(np.max(preds) * 100))[:6] + "%"]

    return jsonify({'predictions': res})


if __name__ == '__main__':
    app.run(debug=False)
