from PIL import Image
import numpy as np
from flask import Flask, request

app = Flask(__name__)
 
@app.route('/')
def hello_world():
    return "<h1>Welcome</h1>"

@app.route('/upload', methods=['POST', 'GET'])
def upload():
    image = request.files['gambar']
    img = Image.open(image)
    img.resize((300,300))
    img_array = np.array(img)
    img.close()
    return "<h1>Upload succesful</h1>"