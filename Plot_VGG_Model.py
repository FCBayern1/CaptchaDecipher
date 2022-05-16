# -*- coding: utf-8 -*-
# @Time    : 2022/5/4 13:46
# @Author  : Joshua
# @FileName: Plot_VGG_Model.py
# @Software: PyCharm
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
HISTORY_DIR = './history/train_demo/' # the directory of the pre-trained model
history = {}
for filename in glob.glob(HISTORY_DIR + '*.history'):
    with open(filename, 'rb') as f:
        history[filename] = pickle.load(f)
# visualization method
def plot_training(history=None, metric='accuracy', title='Model Accuracy', loc='lower right'):
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

for key, val in history.items():
    print(key.replace(HISTORY_DIR, '').rstrip('.history'), val.keys())

plot_training(history)
plot_training(history, metric='loss', title='Model Loss', loc='upper right')
plot_training(history, metric='val_accuracy', title='Model Accuracy (val)')
plot_training(history, metric='val_loss', title='Model Loss (val)', loc='upper right')