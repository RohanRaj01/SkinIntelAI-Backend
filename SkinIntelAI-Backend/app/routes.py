from flask import Blueprint, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

main = Blueprint('main', __name__)
model = load_model('app/skin_cancer_model.h5')
class_labels = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']

@main.route('/')
def home():
    return render_template('index.html')

@main.route('/predict', methods=['POST'])
def predict():
    img_file = request.files['image']
    path = os.path.join('app/static/uploads', img_file.filename)
    img_file.save(path)

    img = image.load_img(path, target_size=(64, 64))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)[0]
    pred_idx = np.argmax(preds)

    return jsonify({
        'label': class_labels[pred_idx],
        'confidence': float(preds[pred_idx] * 100),
        'probabilities': {cls: float(prob * 100) for cls, prob in zip(class_labels, preds)}
    })
from flask import Blueprint, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

main = Blueprint('main', __name__)
model = load_model('app/skin_cancer_model.h5')
class_labels = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']

@main.route('/')
def home():
    return render_template('index.html')

@main.route('/predict', methods=['POST'])
def predict():
    img_file = request.files['image']
    path = os.path.join('app/static/uploads', img_file.filename)
    img_file.save(path)

    img = image.load_img(path, target_size=(64, 64))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)[0]
    pred_idx = np.argmax(preds)

    return jsonify({
        'label': class_labels[pred_idx],
        'confidence': float(preds[pred_idx] * 100),
        'probabilities': {cls: float(prob * 100) for cls, prob in zip(class_labels, preds)}
    })
