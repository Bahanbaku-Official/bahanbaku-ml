from PIL import Image
import numpy as np
from flask import Flask, jsonify, request
from tensorflow.keras.models import load_model

app = Flask(__name__)

food_classifier = load_model('../model_25_classes',compile = False)
food_list = ["apple_pie","bakso","bibimbap","bread_pudding","cheesecake","chicken_curry","chicken_wings","chocolate_cake","french_fries","gado","garlic_bread","gnocchi","gudeg","hamburger","omelette","pizza","rendang","samosa","sate","shrimp_and_grits","strawberry_shortcake","tacos","tiramisu","tuna_tartare","waffles"]

@app.route('/')
def hello_world():
    return "<h1>Welcome</h1>"

@app.route('/upload', methods=['POST', 'GET'])
def upload():
    image = request.files['gambar']
    img = Image.open(image)

    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)

    pred = food_classifier.predict(img_array)
    food_list.sort()
    top_n = 3
    indices = np.argpartition(pred, -top_n)[-top_n:]
    indices = np.squeeze(indices)
    indices = np.flip(indices)

    response = {
        [
            {
                'food': food_list[indices[0]],
                'probability': pred[0][indices[0]]
            },
            {
                'food': food_list[indices[1]],
                'probability': pred[0][indices[1]]
            },
            {
                'food': food_list[indices[2]],
                'probability': pred[0][indices[2]]
            }
        ]
    }

    img.close()
    return jsonify(response)