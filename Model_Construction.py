# -*- coding: utf-8 -*-
# @Time    : 2022/1/6 9:52
# @Author  : Joshua
# @FileName: Model_Construction.py
# @Software: PyCharm

from PIL import Image
from keras import backend as Keras_Backend
from keras.utils.vis_utils import plot_model
from keras.models import *
from keras.layers import *
import glob as glob
import pickle as pickle
import numpy as np
from matplotlib import pyplot as plt
from skimage.filters import threshold_niblack
import tensorflow as tf
import imgaug.augmenters as iaa
from tensorflow.python.platform import gfile
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adagrad
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


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
TRAIN_DATASET_DIR = ".\\train-data\\" # directory of training data set
TEST_DATASET_DIR = ".\\test-data\\"   # directory of test data set

# Define the keys in the model training
BATCH_SIZE = 100
EPOCHS = 100
OPT = 'RMSprop'                        #use adam optimizer
LOSS = 'binary_crossentropy'        #use binaryCE loss function

# define the directory and format of the model saving
MODEL_DIR = './model/train_demo/'
MODEL_FORMAT = '.h5'
HISTORY_DIR = './history/train_demo/'
HISTORY_FORMAT = '.history'

filename_str = "{}captcha_{}_{}_bs_{}_epochs_{}{}"

# Model network structure picture address
MODEL_VIS_FILE = 'captcha_classfication' + '.png'
# Model file
MODEL_FILE = filename_str.format(MODEL_DIR, OPT, LOSS, str(BATCH_SIZE), str(EPOCHS), MODEL_FORMAT)
# training history file
HISTORY_FILE = filename_str.format(HISTORY_DIR, OPT, LOSS, str(BATCH_SIZE), str(EPOCHS), HISTORY_FORMAT)

# graying
def rgb2gray(img):
    # Y' = 0.299 R + 0.587 G + 0.114 B
    # https://en.wikipedia.org/wiki/Grayscale#Converting_color_to_grayscale
    return np.dot(img[...,:3], [0.299, 0.587, 0.114])

# Binarization
def binarify(img):
    thresh_niblack = threshold_niblack(img, window_size=25, k=0.40)
    binary_niblack = img > thresh_niblack
    return(binary_niblack)

# one-hot coding
# Define the one-hot coding function
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

# Vector to text function
def vec2text(vector):
    if not isinstance(vector, np.ndarray):
        vector = np.asarray(vector)
    vector = np.reshape(vector, [CAPTCHA_LENGTH, -1])
    text = ''
    for item in vector:
        text += CAPTCHA_CHARSET[np.argmax(item)]
    return text


# Fit to keras image data format
def fit_keras_channels(batch, rows=CAPTCHA_HEIGHT, cols=CAPTCHA_WIDTH):
    if Keras_Backend.image_data_format() == 'channels_first':
        batch = batch.reshape(batch.shape[0], 1, rows, cols)
        input_shape = (1, rows, cols)
    else:
        batch = batch.reshape(batch.shape[0], rows, cols, 1)
        input_shape = (rows, cols, 1)

    return batch, input_shape


# define and read the training set
X_train = []
Y_train = []
for filename in glob.glob(TRAIN_DATASET_DIR + "*.png"):
    X_train.append(np.array(Image.open(filename)))
    # Get the text value of the actual verification code,label
    Y_train.append(filename.lstrip(TRAIN_DATASET_DIR).rstrip(".png"))

# plt.figure()
# plt.imshow(X_train[0])
# plt.show()
# print(Y_train[0])

# process the images(graying)
# list -> rgb(numpy)
X_train = np.array(X_train, dtype=np.float32)
# rgb -> gray
X_train = rgb2gray(X_train)
# Binarization
# normalization
X_train = X_train / 255
# Fit keras channels
X_train, input_shape = fit_keras_channels(X_train)

# print(X_train.shape, type(X_train))
# print(input_shape)

# one—hot coding the training data
Y_train = list(Y_train)

for i in range(len(Y_train)):
    Y_train[i] = text2vec(Y_train[i])

Y_train = np.asarray(Y_train)

# print(Y_train.shape, type(Y_train))
# print(Y_train[0])


# define and read the test set
X_test = []
Y_test = []

for filename in glob.glob(TEST_DATASET_DIR + "*.png"):
    X_test.append(np.array(Image.open(filename)))
    # Get the text value of the actual verification code,label
    Y_test.append(filename.lstrip(TEST_DATASET_DIR).rstrip("*.png"))

# list->image array numpy(rgb)->graying->normalization（/255）->fit to keras image data format
X_test = np.array(X_test,dtype=np.float32)
X_test = rgb2gray(X_test)
X_test = X_test / 255
X_test,_ = fit_keras_channels(X_test)

# one-hot coding to the test data
Y_test = list(Y_test)
for i in range(len(Y_test)):
    Y_test[i] = text2vec(Y_test[i])

Y_test = np.asarray(Y_test)

print(X_test.shape,type(X_test))
print(Y_test.shape,type(Y_test))
# Construct the model
# input layer
# First conv layer
# tf.nn.conv2d(input,filter, strides, padding, use_cudnn_on_gpu=bool, data_format，name=None)

# 输入层
inputs = Input(shape = input_shape,name = "inputs")

# 第1层卷积
conv1 = Conv2D(32,(3,3),name="conv1")(inputs)
# bn1 = BatchNormalization()(conv1)
relu1 = Activation("relu",name="relu1")(conv1)
# pool1 = MaxPooling2D(pool_size=(2,2),padding="same",name="pool1")(relu1)

# 第2层卷积
conv2 = Conv2D(32,(3,3),name ="conv2")(relu1)
# bn2 = BatchNormalization()(conv2)
relu2 = Activation("relu",name="relu2")(conv2)
pool2 = MaxPooling2D(pool_size=(2,2),padding="same",name="pool2")(relu2)

# 第3层卷积
conv3 = Conv2D(64,(3,3),name="conv3")(pool2)
relu3 = Activation("relu",name="relu3")(conv3)
pool3 = MaxPooling2D(pool_size=(2,2),padding="same",name="pool3")(relu3)

# 将Pooled feature map 摊平后输入全连接网络
x = Flatten()(pool3)

# Dropout
x = Dropout(0.35)(x)

# x = BatchNormalization()(x)

# 4个全连接层分别做10分类，分别对应4个字符
x = [Dense(10,activation="softmax",name="fc%d"%(i+1))(x) for i in range(4)]

# 4个字符向量拼接在一起，与标签向量形式一致，作为模型输出
outs = Concatenate()(x)

# 定义模型的输入与输出
model = Model(inputs=inputs,outputs=outs)
reduce_lr = ReduceLROnPlateau(patience =3, factor = 0.5,verbose = 1)
model.compile(
    optimizer=OPT,
    loss = LOSS,
    metrics = ['acc']
)

print(model.summary())


# Model visualization
plot_model(model,to_file=MODEL_VIS_FILE,show_shapes=True)

earlystopping = EarlyStopping(monitor ="val_loss",
                             mode ="min", patience = 10,
                              min_delta = 1e-4,
                             restore_best_weights = True)
# Training model
# X for input_features Y for labels
history = model.fit(X_train,
                   Y_train,
                   batch_size=BATCH_SIZE,
                   epochs=EPOCHS,
                   verbose=1,
                   validation_data=(X_test,Y_test))



if not gfile.Exists(MODEL_DIR):
    gfile.MakeDirs(MODEL_DIR)

model.save(MODEL_FILE)
print("saved trained model at %s" %MODEL_FILE)

if gfile.Exists(HISTORY_DIR) == False:
    gfile.MakeDirs(HISTORY_DIR)

with open(HISTORY_FILE,"wb") as f:
    pickle.dump(history.history,f)