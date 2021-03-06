from os import link
from PIL import Image
import numpy as np
from flask import Flask, jsonify, request
from tensorflow.keras.models import load_model

app = Flask(__name__)

FOOD_CLASSIFIER = load_model('api/model_25_classes',compile = False) # need to load checkpoint_25_classes folder from google drive to VM first
FOOD_NAMES = []
FOOD_IMG_LINKS = []

with open('api/app/food-mapping.txt','r') as data:
    name_and_links = data.readlines()
    name_and_links.sort()
    FOOD_NAMES = [food.strip().split('|')[0] for food in name_and_links]
    FOOD_IMG_LINKS = [food.strip().split('|')[1] for food in name_and_links]

@app.route('/')
def hello_world():
    return "<h1>Welcome</h1>"

@app.route('/upload', methods=['POST', 'GET'])
def upload():
    image = request.files['gambar']
    img = Image.open(image)
    img = img.resize((150,150))

    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)

    pred = FOOD_CLASSIFIER.predict(img_array)
    FOOD_NAMES.sort()
    top_n = 3
    indices = np.argpartition(pred, -top_n)[-top_n:]
    indices = np.squeeze(indices)
    indices = np.flip(indices)

    response_list = [
        {
            'food': FOOD_NAMES[indices[0]].replace('_', ' '),
            'probability': pred[0][indices[0]],
            'image_link': FOOD_IMG_LINKS[indices[0]]
        },
        {
            'food': FOOD_NAMES[indices[1]].replace('_', ' '),
            'probability': pred[0][indices[1]],
            'image_link': FOOD_IMG_LINKS[indices[1]]
        },
        {
            'food': FOOD_NAMES[indices[2]].replace('_', ' '),
            'probability': pred[0][indices[2]],
            'image_link': FOOD_IMG_LINKS[indices[2]]
        }
    ]

    response_list.sort(key=lambda obj : obj['probability'], reverse=True)

    response_list = [{ 'food': item['food'], 'probability': str(item['probability']), 'image_link': item['image_link']} for item in response_list]

    response_obj = { 'result': response_list }

    img.close()
    
    return jsonify(response_obj)

app.run()