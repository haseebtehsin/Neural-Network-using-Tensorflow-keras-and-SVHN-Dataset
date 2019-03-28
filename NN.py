from __future__ import absolute_import, division, print_function

import scipy.io as spio
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
# Helper libraries
import numpy as np


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


# ----- Load training and test data ------#
mat1 = spio.loadmat('extra_32x32.mat', squeeze_me=True)     # extra data used
mat2 = spio.loadmat('test_32x32.mat', squeeze_me=True)
train_images=mat1['X']
train_labels=mat1['y']
test_images=mat2['X']
test_labels=mat2['y']

# ---- Specify class labels ----#
class_names = ['0','1', '2', '3', '4', '5','6', '7', '8', '9']


# ------------- Convert to proper format -------------#
train_images=formatArray(train_images)
test_images=formatArray(test_images)
train_labels=fixLabel(train_labels)
test_labels=fixLabel(test_labels)


# ------------- Normalize ---------------#
train_images = train_images / 255.0
test_images = test_images / 255.0

# --------- Create Training Model, specify layers ----#
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(32,32)),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.23),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

# ------ Compile model with optimizer, loss functions and metrics ---- #
model.compile(optimizer='Adadelta',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# ----- Apply model on training data with 6 epochs ---#
model.fit(train_images, train_labels, epochs=6)

# ---- Save model for future use ----#
model.save('my_model.h5')

# ---- get metrics ---- #
test_loss, test_acc = model.evaluate(test_images, test_labels)

# ---- Print metrics ----#
print('Test accuracy:', test_acc)





