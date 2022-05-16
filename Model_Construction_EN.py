# -*- coding: utf-8 -*-
# @Time    : 2022/5/1 10:53
# @Author  : Joshua
# @FileName: Model_Construction_EN.py
# @Software: PyCharm
from PIL import Image
from keras import backend as Keras_Backend
from keras.utils.vis_utils import plot_model
from keras.models import *
from keras.layers import *
from tensorflow.python.keras.optimizers import *
import glob as glob
import pickle as pickle
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from tensorflow.python.platform import gfile

# Define the superkeys
NUMBER=['0','1','2','3','4','5','6','7','8','9']

LOWERCASE = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
            'v', 'w', 'x', 'y', 'z']

UPPERCASE = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
           'V', 'W', 'X', 'Y', 'Z']

CAPTCHA_CHARSET = UPPERCASE   # Captcha's charset
CAPTCHA_LENGTH = 4            # Length of the captcha image
CAPTCHA_HEIGHT = 60        # Height of the captcha image
CAPTCHA_WIDTH = 160        # Width of the captcha image

TRAIN_DATASET_SIZE = 20000      # Size of training data set
TEST_DATASET_SIZE = 2000        # Size of test data set
TRAIN_DATASET_DIR = ".\\train-data_EN\\" # directory of training data set
TEST_DATASET_DIR = ".\\test-data_EN\\"   # directory of test data set

# Define the keys in the model training
BATCH_SIZE = 100
EPOCHS = 10
OPT = 'adam'                       #use adam optimizer
LOSS = 'binary_crossentropy'        #use binaryCE loss function

# define the directory and format of the model saving
MODEL_DIR = './model/train_demo_EN/'
MODEL_FORMAT = '.h5'
HISTORY_DIR = './history/train_demo_EN/'
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

# one-hot coding
# Define the one-hot coding function
def text2vec(text,length=CAPTCHA_LENGTH,charset=CAPTCHA_CHARSET):
    # define the length of text
    text_len = len(text)
    # validation of the legitimacy of the captcha
    if text_len != length:
        raise ValueError("Error:length of captcha should be {},but got {}".format(length,text_len))
    # Generate a one-dimensional vector shaped like (captcha_len * captcha_charset)
    # For example, four pure digital verification codes generate a one-dimensional vector in the shape of (4 * 10,)
    vec = np.zeros(length*len(charset))
    for i in range(length):
        # Use one-hot coding function to operate each number in the captcha
        vec[ord(text[i])-65 + i*26] = 1
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
    Y_train.append(filename.replace(TRAIN_DATASET_DIR,'').rstrip(".png"))

# process the images(graying)
# list -> rgb(numpy)
X_train = np.array(X_train, dtype=np.float32)
# rgb -> gray
X_train = rgb2gray(X_train)
# normalization
X_train = X_train / 255
# Fit keras channels
X_train, input_shape = fit_keras_channels(X_train)

# one—hot coding the training data
Y_train = list(Y_train)
for i in range(len(Y_train)):
    Y_train[i] = text2vec(Y_train[i])

Y_train = np.asarray(Y_train)

print(Y_train.shape, type(Y_train))
print(Y_train[0])

# define and read the test set
X_test = []
Y_test = []

for filename in glob.glob(TEST_DATASET_DIR + "*.png"):
    X_test.append(np.array(Image.open(filename)))
    # Get the text value of the actual verification code,label
    Y_test.append(filename.replace(TEST_DATASET_DIR,'').rstrip("*.png"))

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

# print(X_test.shape,type(X_test))
# print(Y_test.shape,type(Y_test))


# Construct the model
# input layer

inputs = Input(shape = input_shape,name = "inputs")

# First conv layer
# tf.nn.conv2d(input,filter, strides, padding, use_cudnn_on_gpu=bool, data_format，name=None)
# First round
conv1 = Conv2D(64,(3,3),name="conv1" ,kernel_initializer='he_uniform')(inputs)
bn1=BatchNormalization()(conv1)
relu1 = Activation("relu",name="relu1")(bn1)

conv2 = Conv2D(64,(3,3),name ="conv2", kernel_initializer='he_uniform')(relu1)
bn2=BatchNormalization()(conv2)
relu2 = Activation("relu",name="relu2")(bn2)

pool1=MaxPooling2D(pool_size=(2,2),padding="same",name="pool1")(relu2)

# Second round
conv3= Conv2D(128, (3, 3), name='conv3',padding='same', kernel_initializer='he_uniform')(pool1)
bn3=BatchNormalization()(conv3)
relu3=Activation("relu",name="relu3")(bn3)

conv4=Conv2D(128, (3, 3), name='conv4',padding='same', kernel_initializer='he_uniform')(relu3)
bn4=BatchNormalization()(conv4)
relu4=Activation("relu",name="relu4")(bn4)

pool2 = MaxPooling2D(pool_size=(2, 2), padding='same', name='pool2')(relu4)

# Third round
conv5 = Conv2D(256, (3,3), name='conv5',padding='same', kernel_initializer='he_uniform')(pool2)
bn5 = BatchNormalization()(conv5)
relu5 = Activation('relu',name="relu5")(bn5)
# drop3 = Dropout(0.4)(act5)

conv6 = Conv2D(256, (3, 3), name='conv6',padding='same', kernel_initializer='he_uniform')(relu5)
bn6 = BatchNormalization()(conv6)
relu6 = Activation('relu',name="relu6")(bn6)
# drop4 = Dropout(0.4)(act6)

conv7 = Conv2D(256, (3, 3), name='conv7',padding='same', kernel_initializer='he_uniform')(relu6)
bn7 = BatchNormalization()(conv7)
relu7 = Activation('relu',name="relu7")(bn7)

pool3 = MaxPooling2D(pool_size=(2, 2), padding='same', name='pool3')(relu7)

# Fourth round
conv8 = Conv2D(512, (3, 3), name='conv8',padding='same', kernel_initializer='he_uniform')(pool3)
bn8 = BatchNormalization()(conv8)
relu8 = Activation('relu',name="relu8")(bn8)
# drop5 = Dropout(0.4)(act8)

conv9 = Conv2D(512, (3, 3), name='conv9',padding='same', kernel_initializer='he_uniform')(relu8)
bn9 = BatchNormalization()(conv9)
relu9 = Activation('relu',name="relu9")(bn9)
# drop6 = Dropout(0.4)(act9)

conv10 = Conv2D(512, (3, 3), name='conv10',padding='same', kernel_initializer='he_uniform')(relu9)
bn10 = BatchNormalization()(conv10)
relu10 = Activation('relu',name="relu10")(bn10)

pool4 = MaxPooling2D(pool_size=(2, 2), padding='same', name='pool4')(relu10)

# fifth round
conv11 = Conv2D(512, (3, 3), name='conv11', padding='same', kernel_initializer='he_uniform')(pool4)
bn11 = BatchNormalization()(conv11)
relu11 = Activation('relu',name="relu11")(bn11)
# drop7 = Dropout(0.4)(act11)

conv12 = Conv2D(512, (3, 3), name='conv12', padding='same', kernel_initializer='he_uniform')(relu11)
bn12 = BatchNormalization()(conv12)
relu12 = Activation('relu',name="relu12")(bn12)
# drop8 = Dropout(0.4)(act12)

conv13 = Conv2D(512, (3, 3), name='conv13', padding='same', kernel_initializer='he_uniform')(relu12)
bn13 = BatchNormalization()(conv13)
relu13 = Activation('relu',name="relu13")(bn13)

pool5 = MaxPooling2D(pool_size=(2, 2), padding='same', name='pool5')(relu13)

# full connection layer
x = Flatten()(pool5)

# dropout layer
x = Dropout(0.5)(x)

x1 = Dense(4096)(x)
bnx1 = BatchNormalization()(x1)
actx1 = Activation('relu')(bnx1)
drop9 = Dropout(0.4)(actx1)

x2 = Dense(4096)(drop9)
bnx2 = BatchNormalization()(x2)
x = Activation('relu')(bnx2)

# Softmax and classification by 4 full-connection layer
x = [Dense(26,activation="softmax",name="fc%d"%(i+1))(x) for i in range(4)]

# The four character vectors are spliced together, consistent with the form of label vector, and output as a model
outs = Concatenate()(x)

# Define the input and output of the model
model = Model(inputs=inputs,outputs=outs)
model.compile(optimizer=OPT,loss=LOSS,metrics=["accuracy"])

print(model.summary())


# Model visualization
# plot_model(model,to_file=MODEL_VIS_FILE,show_shapes=True)

# Training model
# X for input_features Y for labels
history = model.fit(X_train,
                   Y_train,
                   batch_size=BATCH_SIZE,
                   epochs=EPOCHS,
                   verbose=2,
                   validation_data=(X_test,Y_test))



if not gfile.Exists(MODEL_DIR):
    gfile.MakeDirs(MODEL_DIR)

model.save(MODEL_FILE)
print("saved trained model at %s" %MODEL_FILE)

if gfile.Exists(HISTORY_DIR) == False:
    gfile.MakeDirs(HISTORY_DIR)

with open(HISTORY_FILE,"wb") as f:
    pickle.dump(history.history,f)




