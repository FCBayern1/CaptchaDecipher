# -*- coding: utf-8 -*-
# @Time    : 2022/3/29 17:46
# @Author  : Joshua
# @FileName: CaptchaVisualize.py
# @Software: PyCharm
import random
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image
from captcha.image import ImageCaptcha

# Define the superkeys
NUMBER=['0','1','2','3','4','5','6','7','8','9']

LOWERCASE = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
            'v', 'w', 'x', 'y', 'z']

UPPERCASE = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
           'V', 'W', 'X', 'Y', 'Z']

CAPTCHA_CHARSET = NUMBER   # Captcha's charset
CAPTCHA_LEN = 4            # Length of the captcha image
CAPTCHA_HEIGHT = 60        # Height of the captcha image
CAPTCHA_WIDTH = 160        # Width of the captcha image

TRAIN_DATASET_SIZE = 10000      # Size of training data set
TEST_DATASET_SIZE = 2000        # Size of test data set
TRAIN_DATA_DIR = "./train-data/" # directory of training data set
TEST_DATA_DIR = "./test-data/"   # directory of test data set

# Generate the random characters
def gen_random_text(charset=CAPTCHA_CHARSET,length=CAPTCHA_LEN):
    text = [random.choice(charset) for a in range(length)]
    return "".join(text)

def gen_captcha_dataset(size=100, height=60, width=160, image_font=".png"):
    # 创建ImageCaptcha实例captcha
    captcha = ImageCaptcha(width=width, height=height)

    # 创建图像和文本数组
    images =[None] * size
    texts =[None] * size
    for i in range(size):
        # Generate random verification code characters
        texts[i] = gen_random_text(CAPTCHA_CHARSET, CAPTCHA_LEN)
        # Use PIL.Image.open() to identify the newly generated verification code image
        # Then, the image is converted into a numpy array shaped like (CAPTCHA_WIDTH, CAPTCHA_HEIGHT, 3)
        images[i] = np.array(Image.open(captcha.generate(texts[i])))

    return images, texts
def rgb2gray(img):
    # We introduce the formula Y' = 0.299 R + 0.587 G + 0.114 B
    return np.dot(img[...,:3],[0.299,0.587,0.114])
# convertion
# Generate and return images and labels
images,texts = gen_captcha_dataset()


# image visualization
    # define the figure
plt.figure()
for i in range(20):
    plt.subplot(5,4,i+1) # Draw the first 20 verification codes and display them in the form of 5 rows and 4 columns of subgraphs
    plt.tight_layout()   # format and size auto-fitting
    plt.imshow(images[i])
    plt.title("Label:{}".format(texts[i])) # Set label as sub graph title
    plt.xticks([])       # delete x axis label
    plt.yticks([])       # delete y axis label
plt.show()
