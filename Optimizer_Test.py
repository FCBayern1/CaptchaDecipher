# -*- coding: utf-8 -*-
# @Time    : 2022/4/18 18:25
# @Author  : Joshua
# @FileName: Optimizer_Test.py
# @Software: PyCharm
import glob
import pickle

import numpy as np
import matplotlib.pyplot as plt
history_file = './pre-trained/history/optimizer/binary_ce/captcha_adam_binary_crossentropy_bs_100_epochs_100.history'
HISTORY_DIR = './model_history/optimizer/'
HISTORY_DIR_adam = './model_history/optimizer_adam/'
with open(history_file,"rb") as f:
    history = pickle.load(f)

# fig = plt.figure()
# plt.subplot(2,1,1)
# plt.plot(history['acc'])
# plt.plot(history['val_acc'])
# plt.title('Model Accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='lower right')
#
# plt.subplot(2,1,2)
# plt.plot(history['loss'])
# plt.plot(history['val_loss'])
# plt.title('Model Loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper right')
# plt.tight_layout()
#
# plt.show()

def plot_training(history=None, metric='acc', title='Model Accuracy', loc='lower right'):
    model_list = []
    fig = plt.figure(figsize=(10, 8))
    for key, val in history.items():

        model_list.append(key.replace(HISTORY_DIR, '').rstrip('.history'))
        plt.plot(val[metric])

    plt.title(title)
    plt.ylabel(metric)
    plt.xlabel('epoch')
    plt.legend(model_list, loc=loc)
    plt.show()


history = {}
history_adam = {}
for filename in glob.glob(HISTORY_DIR + '*.history'):
    with open(filename, 'rb') as f:
        history[filename] = pickle.load(f)

for filename in glob.glob(HISTORY_DIR_adam + '*.history'):
    with open(filename, 'rb') as f:
        history_adam[filename] = pickle.load(f)

for key, val in history.items():
    print(key.replace(HISTORY_DIR, '').rstrip('.history'), val.keys())


plot_training(history)
plot_training(history, metric='loss', title='Model Loss', loc='upper right')
plot_training(history, metric='val_acc', title='Model Accuracy (val)')
plot_training(history, metric='val_loss', title='Model Loss (val)', loc='upper right')

