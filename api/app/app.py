from PIL import Image
import numpy as np
from flask import Flask, request

app = Flask(__name__)
 
@app.route('/', methods=['POST', 'GET'])
def hello_world():
    image = request.files['gambar']
    img = Image.open(image)
    img.resize((300,300))
    img_array = np.array(img)
    img.close()
    return "data"
