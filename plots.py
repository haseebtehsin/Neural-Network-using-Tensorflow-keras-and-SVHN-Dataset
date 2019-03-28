from __future__ import absolute_import, division, print_function

import scipy.io as spio
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt


# ---- Function to convert rgb images to grayscale -----#
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


# ---- Function to format data to use with keras ----#
def formatArray(data):
    im = []
    for i in range(0, data.shape[3]):
        im.append(rgb2gray(data[:, :, :, i]))
    return np.asarray(im)


# ---- Replace 10 in labels with 0 ----#
def fixLabel(labels):
    labels[labels == 10] = 0
    return labels


# ---- Function for plotting an image ---#
def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]),
               color=color)


# ---- Function to plot prediction value --- #
def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


# ----- Load test data ------#
mat2 = spio.loadmat('test_32x32.mat', squeeze_me=True)
test_images=mat2['X']
test_labels=mat2['y']

# ---- Specify class labels ----#
class_names = ['0','1', '2', '3', '4', '5','6', '7', '8', '9']


# ------------- Convert to proper format -------------#
test_images=formatArray(test_images)
test_labels=fixLabel(test_labels)

# ------------- Normalize ---------------#
test_images = test_images / 255.0

# ---- Load the trained Model ---- #
model= keras.models.load_model('my_model.h5')

# ---- Save predictions by applying model ---- #
predictions = model.predict(test_images)

# --- Plot the desired images ----#
num_rows = 3
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions, test_labels, test_images)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions, test_labels)
plt.show()




