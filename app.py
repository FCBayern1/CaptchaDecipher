import base64

import numpy as np
import tensorflow as tf

from io import BytesIO
from flask import Flask, request, jsonify
from keras.models import load_model
from PIL import Image
from skimage.filters import threshold_niblack
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


NUMBER = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

LOWERCASE = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
            'v', 'w', 'x', 'y', 'z']

UPPERCASE = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
           'V', 'W', 'X', 'Y', 'Z']

CAPTCHA_CHARSET = NUMBER   # Captcha set
CAPTCHA_CHARSET_EN=UPPERCASE
CAPTCHA_LENGTH = 4         # Length of the captcha
CAPTCHA_HEIGHT = 60        # Height of the captcha
CAPTCHA_WIDTH = 160        # Width of the captcha

# The directory of the saved model
MODEL_FILE = './model/train_demo/captcha_rmsprop_binary_crossentropy_bs_100_epochs_100.h5'
MODEL_FILE_EN='./model/train_demo_EN/captcha_adam_binary_crossentropy_bs_100_epochs_10.h5'
MODEL_FILE_RMSPROP='./pre-trained/model/captcha_rmsprop_binary_crossentropy_bs_100_epochs_10.h5'

# Vector to text
def vec2text(vector):
    if not isinstance(vector, np.ndarray):
        vector = np.asarray(vector)
    vector = np.reshape(vector, [CAPTCHA_LENGTH, -1])
    text = ''
    for item in vector:
        text += CAPTCHA_CHARSET[np.argmax(item)]
    return text

def vec2text_EN(vector):
    if not isinstance(vector, np.ndarray):
        vector = np.asarray(vector)
    vector = np.reshape(vector, [CAPTCHA_LENGTH, -1])
    text = ''
    for item in vector:
        text += CAPTCHA_CHARSET_EN[np.argmax(item)]
    return text
# graying method
def rgb2gray(img):
    # Y' = 0.299 R + 0.587 G + 0.114 B
    # https://en.wikipedia.org/wiki/Grayscale#Converting_color_to_grayscale
    return np.dot(img[...,:3], [0.299, 0.587, 0.114])

def binarify(img):
    thresh_niblack = threshold_niblack(img, window_size=25, k=0.40)
    binary_niblack = img > thresh_niblack
    return(binary_niblack)

app = Flask(__name__) # create the instance of flask

# Ping-Pong URL
@app.route('/ping', methods=['GET', 'POST'])
def hello_world():
    return 'pong'

# identification URL
@app.route('/predict', methods=['POST'])
def predict():
    response = {'success': False, 'prediction': '', 'debug': 'error'}
    received_image= False
    if request.method == 'POST':
        if request.files.get('image'): # read the image files
            image = request.files['image'].read()
            received_image = True
            response['debug'] = 'get image'
        elif request.get_json(): # base64 encoding the image files
            encoded_image = request.get_json()['image']
            image = base64.b64decode(encoded_image)
            received_image = True
            response['debug'] = 'get json'
        if received_image:
            image = np.array(Image.open(BytesIO(image)))
            image = rgb2gray(image).reshape(1, 60, 160, 1).astype('float32') / 255
            pred = model.predict(image)
            response['prediction'] = response['prediction'] + vec2text(pred)
            response['success'] = True
            response['debug'] = 'predicted'
    else:
        response['debug'] = 'No Post'
    return jsonify(response)

# letter identification URL
@app.route('/predict_EN', methods=['POST'])
def predict_EN():
    response = {'success': False, 'prediction': '', 'debug': 'error'}
    received_image= False
    if request.method == 'POST':
        if request.files.get('image'): # read the image files
            image = request.files['image'].read()
            received_image = True
            response['debug'] = 'get image'
        elif request.get_json(): # base64 encoding the image files
            encoded_image = request.get_json()['image']
            image = base64.b64decode(encoded_image)
            received_image = True
            response['debug'] = 'get json'
        if received_image:
            image = np.array(Image.open(BytesIO(image)))
            image = rgb2gray(image).reshape(1, 60, 160, 1).astype('float32') / 255
            pred = model_EN.predict(image)
            response['prediction'] = response['prediction'] + vec2text_EN(pred)
            response['success'] = True
            response['debug'] = 'predicted'
    else:
        response['debug'] = 'No Post'
    return jsonify(response)

model = load_model(MODEL_FILE) # read the model
model_EN=load_model(MODEL_FILE_EN) #read the model for letter identification


if __name__ == "__main__":
    app.run()
