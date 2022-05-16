import keras.models
from PIL import Image
from keras import backend as K
import glob
import numpy as np
from keras.utils.vis_utils import plot_model

UPPERCASE = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
           'V', 'W', 'X', 'Y', 'Z']
TEST_DATA_DIR = '.\\test-data_EN\\' # test data directory
CAPTCHA_CHARSET = UPPERCASE   # captcha charset
CAPTCHA_LEN = 4            # length of captcha
CAPTCHA_HEIGHT = 60        # height of captcha
CAPTCHA_WIDTH = 160        # width of captcha
BATCH_SIZE = 100
EPOCHS = 10
OPT = 'adam'
LOSS = 'binary_crossentropy'
MODEL_DIR = './model/train_demo_EN/'
MODEL_FORMAT = '.h5'
filename_str = "{}captcha_{}_{}_bs_{}_epochs_{}{}"
# model file
MODEL_FILE = filename_str.format(MODEL_DIR, OPT, LOSS, str(BATCH_SIZE), str(EPOCHS), MODEL_FORMAT)
MODEL_VIS_FILE = 'captcha_classfication_EN' + '.png'

# graying
def rgb2gray(img):
    # Y' = 0.299 R + 0.587 G + 0.114 B
    return np.dot(img[...,:3], [0.299, 0.587, 0.114])

# adapt to keras image format
def fit_keras_channels(batch, rows=CAPTCHA_HEIGHT, cols=CAPTCHA_WIDTH):
    if K.image_data_format() == 'channels_first':
        batch = batch.reshape(batch.shape[0], 1, rows, cols)
        input_shape = (1, rows, cols)
    else:
        batch = batch.reshape(batch.shape[0], rows, cols, 1)
        input_shape = (rows, cols, 1)

    return batch, input_shape


# one-hot coding
def text2vec(text,length=4,charset=CAPTCHA_CHARSET):
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
        # Hotcode per character = index + offset
        vec[ord(text[i])-65 + i*26] = 1
    return vec

# Vector to text function
def vec2text(vector):
    if not isinstance(vector, np.ndarray):
        vector = np.asarray(vector)
    vector = np.reshape(vector, [4, -1])
    text = ''
    for item in vector:
        text += CAPTCHA_CHARSET[np.argmax(item)]
    return text
# read te test dataset
X_test = []
Y_test = []

for filename in glob.glob(TEST_DATA_DIR + "*.png"):
    X_test.append(np.array(Image.open(filename)))
    Y_test.append(filename.replace(TEST_DATA_DIR,'').rstrip(".png"))

# List - > image array numpy (RGB) - > gray graying - > normalization Standardization (/ 255) - >
# fit keras adapt to keras image format
X_test = np.array(X_test,dtype=np.float32)
X_test = rgb2gray(X_test)
X_test = X_test / 255
X_test,_ = fit_keras_channels(X_test)

# one-hot coding
Y_test = list(Y_test)
for i in range(len(Y_test)):
    Y_test[i] = text2vec(Y_test[i])

Y_test = np.asarray(Y_test)
from tensorflow.python.platform import gfile
model=keras.models.load_model(MODEL_FILE)
model.summary()
plot_model(model,to_file=MODEL_VIS_FILE,show_shapes=True)
# actual value
print(vec2text(Y_test[2]))
# predict value
prediction = model.predict(X_test[2].reshape(1,60,160,1))
print(vec2text(prediction))