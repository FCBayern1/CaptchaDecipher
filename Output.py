# -*- coding: utf-8 -*-
# @Time    : 2022/3/30 11:20
# @Author  : Joshua
# @FileName: Output.py
# @Software: PyCharm

import random
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image
import glob
import cv2
from captcha.image import ImageCaptcha
from skimage.filters import (threshold_otsu, threshold_niblack,
                             threshold_sauvola)


# Define the superkeys
NUMBER=['0','1','2','3','4','5','6','7','8','9']

CAPTCHA_CHARSET = NUMBER   # Captcha's charset
CAPTCHA_LEN = 4            # Length of the captcha image
CAPTCHA_HEIGHT = 60        # Height of the captcha image
CAPTCHA_WIDTH = 160        # Width of the captcha image

TRAIN_DATASET_DIR = ".\\train-data\\" # directory of training data set

def rgb2gray(img):
    # Y' = 0.299 R + 0.587 G + 0.114 B
    # https://en.wikipedia.org/wiki/Grayscale#Converting_color_to_grayscale
    return np.dot(img[...,:3], [0.299, 0.587, 0.114])

def binarify(img):
    thresh_niblack = threshold_niblack(img, window_size=25, k=0.40)
    binary_niblack = image > thresh_niblack
    return(binary_niblack)

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

image = rgb2gray(image)
image= binarify(image)

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
