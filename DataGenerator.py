# -*- coding: utf-8 -*-
# @Time    : 2022/3/29 16:49
# @Author  : Joshua
# @FileName: Data_Generator.py
# @Software: PyCharm

import random
from tensorflow.python.platform import gfile
from captcha.image import ImageCaptcha

# Define the superkeys
NUMBER=['0','1','2','3','4','5','6','7','8','9']

LOWERCASE = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
            'v', 'w', 'x', 'y', 'z']

UPPERCASE = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
           'V', 'W', 'X', 'Y', 'Z']

CAPTCHA_CHARSET = NUMBER   # Captcha's charset
CAPTCHA_LENGTH = 4            # Length of the captcha image
CAPTCHA_HEIGHT = 60        # Height of the captcha image
CAPTCHA_WIDTH = 160        # Width of the captcha image

TRAIN_DATASET_SIZE = 10000      # Size of training data set
TEST_DATASET_SIZE = 1000        # Size of test data set
TRAIN_DATASET_DIR = "./train-data/" # directory of training data set
TEST_DATASET_DIR = "./test-data/"   # directory of test data set

# Generate the random characters
def gen_random_text(charset=CAPTCHA_CHARSET,length=CAPTCHA_LENGTH):
    text = [random.choice(charset) for a in range(length)]
    return "".join(text)

# Generate characters array from 0000 to 9999
def gen_training_text(index):
    text=[str(int(index/1000)),str(int((index/100)%10)),str(int((index/10)%10)),str(int(index%10))]
    return "".join(text)

# Create and save the test data set
def create_captcha_dataset(size=100, data_dir="./data/", height=60, width=160, image_format=".png"):
    # if the data_dir is not empty before you save the data, clean the data_dir first
    if gfile.Exists(data_dir):
        gfile.DeleteRecursively(data_dir)
    gfile.MakeDirs(data_dir)

    # Create imagecaptcha instance
    captcha = ImageCaptcha(width=width, height=height)

    for a in range(size):
        # Generate random captcha code characters
        text = gen_random_text(CAPTCHA_CHARSET, CAPTCHA_LENGTH)
        captcha.write(text, data_dir + text + image_format)

    return None

# Create and save the training data set
def create_training_dataset(size=100, data_dir="./data/", height=60, width=160, image_format=".png"):
    # if the data_dir is not empty before you save the data, clean the data_dir first
    if gfile.Exists(data_dir):
        gfile.DeleteRecursively(data_dir)
    gfile.MakeDirs(data_dir)

    # Create imagecaptcha instance
    captcha = ImageCaptcha(width=width, height=height)

    for a in range(size):
        # Generate random captcha code characters
        text = gen_training_text(a)
        captcha.write(text, data_dir + text + image_format)

    return None

# Create and save training sets
# create_training_dataset(TRAIN_DATASET_SIZE,TRAIN_DATASET_DIR)
create_training_dataset(TRAIN_DATASET_SIZE,TRAIN_DATASET_DIR)
# Create and save test sets
create_captcha_dataset(TEST_DATASET_SIZE,TEST_DATASET_DIR)