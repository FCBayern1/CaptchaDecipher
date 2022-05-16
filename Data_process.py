# -*- coding: utf-8 -*-
# @Time    : 2022/1/3 11:23
# @Author  : Joshua
# @FileName: Data_process.py
# @Software: PyCharm

from PIL import Image
from keras import backend as K

import random
import glob
from skimage.filters import threshold_niblack
import numpy as np
from tensorflow.python.platform import gfile
import matplotlib.pyplot as plt

# Define the superkeys
NUMBER=['0','1','2','3','4','5','6','7','8','9']

LOWERCASE = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
            'v', 'w', 'x', 'y', 'z']

UPPERCASE = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
           'V', 'W', 'X', 'Y', 'Z']

CAPTCHA_CHARSET = NUMBER   # Captcha's charset
CAPTCHA_LENGTH = 4         # Length of the captcha image
CAPTCHA_HEIGHT = 60        # Height of the captcha image
CAPTCHA_WIDTH = 160        # Width of the captcha image

TRAIN_DATASET_DIR = ".\\train-data\\" # directory of training data set

# initialization of the keys
image = []
text = []
count = 0

for filename in glob.glob(TRAIN_DATASET_DIR + "*.png"):
    image.append(np.array(Image.open(filename)))
    text.append(filename.lstrip(TRAIN_DATASET_DIR).rstrip(".png"))
    count += 1
    # We only adpats the pre 100 images.
    if count >= 100:
        break

# Visualization of the datas.
plt.figure()
for i in range(20):
    plt.subplot(5,4,i+1) # Draw the first 20 verification codes and display them in the form of
    # 5 rows and 4 columns of subgraphs
    plt.tight_layout()   # Auto fit subgraph size
    plt.imshow(image[i])
    plt.title("Label:{}".format(text[i])) # set the subtitle by the label
    plt.xticks([])       # delete x axis
    plt.yticks([])       # delete y axis
plt.show()
image = np.array(image,dtype=np.float32)

# Define the function of converting to grayscale image
def rgb2gray(img):
    # We introduce the formula Y' = 0.299 R + 0.587 G + 0.114 B
    return np.dot(img[...,:3],[0.299,0.587,0.114])
# convertion

# Binarization
def binarify(img):
    thresh_niblack = threshold_niblack(img, window_size=25, k=0.40)
    binary_niblack = img > thresh_niblack
    return(binary_niblack)

image = rgb2gray(image)


# The graying images
plt.figure()
for i in range(20):
    plt.subplot(5,4,i+1)  # Draw the first 20 verification codes and display them in the form of
    # 5 rows and 4 columns of subgraphs
    plt.tight_layout()    # Auto fit subgraph size
    plt.imshow(image[i],cmap="Greys")
    plt.title("Label:{}".format(text[i])) # set the subtitle by the label
    plt.xticks([])        # delete x axis
    plt.yticks([])        # delete y axis
plt.show()

# Define keras-fitting function
def fit_keras_channels(batch, rows=CAPTCHA_HEIGHT, cols=CAPTCHA_WIDTH):
    if K.image_data_format() == "channels_first":
        batch = batch.reshape(batch.shape[0], 1, rows, cols)
        input_shape = (1, rows, cols)
    else:
        batch = batch.reshape(batch.shape[0], rows, cols, 1)
        input_shape = (rows, cols, 1)

    return batch, input_shape

# Define one-hot coding function
# CAPTCHA_CHARSET = NUMBER   # Charset of captcha
# CAPTCHA_LENGTH = 4            # Length of captcha
def text2vec(text,length=CAPTCHA_LENGTH,charset=CAPTCHA_CHARSET):
    # define the length of text
    text_len = len(text)
    # validation of the legitimacy of the captcha
    if text_len != length:
        raise ValueError("Error:length of captcha should be{},but got {}".format(length,text_len))
    # Generate a one-dimensional vector shaped like (captcha_len * captcha_charset)
    # For example, four pure digital verification codes generate a one-dimensional vector in the shape of (4 * 10,)
    vec = np.zeros(length*len(charset))
    for i in range(length):
        # Use one-hot coding function to operate each number in the captcha
        # Hotcode per character = index + offset
        vec[charset.index(text[i]) + i*len(charset)] = 1
    return vec

text = list(text)
vec = [None]*len(text)

for i in range(len(vec)):
    vec[i] = text2vec(text[i])
