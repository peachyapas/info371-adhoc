#!/usr/bin/env python
# coding: utf-8
# # MNIST Prediction with Full TF Model
## Lauren, Peach || INFO 371 Ad Hoc, with contributions by Ott "Otty" Toomet 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# Tensorflow
import tensorflow as tf
from tensorflow.keras import optimizers
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,     Dropout,Flatten,Dense,Activation,     BatchNormalization
# Image processing
from PIL import Image
import gzip
import os
import re

# set some globals, which are used across the remaining functions
imgWidth = 28
imgHeight = 28
Image_Channels = 1
IMAGE_SHAPE = (imgWidth, imgHeight, Image_Channels)
NUM_CLASSES = 10

# training parameters
EPOCHS = 40
BATCH_SIZE = 128
np.set_printoptions(linewidth=150)

## Load data
def training_images():
    with gzip.open('../data/fullDataset/train-images-idx3-ubyte.gz', 'r') as f:
        # first 4 bytes is a magic number
        magic_number = int.from_bytes(f.read(4), 'big')
        # second 4 bytes is the number of images
        image_count = int.from_bytes(f.read(4), 'big')
        # third 4 bytes is the row count
        row_count = int.from_bytes(f.read(4), 'big')
        # fourth 4 bytes is the column count
        column_count = int.from_bytes(f.read(4), 'big')
        # rest is the image pixel data, each pixel is stored as an unsigned byte
        # pixel values are 0 to 255
        image_data = f.read()
        images = np.frombuffer(image_data, dtype=np.uint8).reshape((image_count, row_count, column_count))
        return images

def training_labels():
    with gzip.open('../data/fullDataset/train-labels-idx1-ubyte.gz', 'r') as f:
        # first 4 bytes is a magic number
        magic_number = int.from_bytes(f.read(4), 'big')
        # second 4 bytes is the number of labels
        label_count = int.from_bytes(f.read(4), 'big')
        # rest is the label data, each label is stored as unsigned byte
        # label values are 0 to 9
        label_data = f.read()
        labels = np.frombuffer(label_data, dtype=np.uint8)
        return labels

images = training_images()
labels = training_labels()
print("Data shape", images.shape)
print("Labels' shape", labels.shape)

# check an image to see what the data structure looks like
i = np.random.choice(len(images), 1)
print("Example image", i, "\n", images[i])
print("It's label:", labels[i])

# convert to float32
images = (images.astype('float32')  / 255.0)
images = images.reshape(len(images), 28, 28, 1) # reshape it to 3D tensor array
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size = 0.2, random_state=1)

# split looks right
print("Training data shape", X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

## Build the model
model = Sequential()

model.add(Conv2D(64, 4, activation='relu',
                 input_shape=IMAGE_SHAPE))
model.add(BatchNormalization())
model.add(Conv2D(64, 4, activation='relu',
                 input_shape=IMAGE_SHAPE))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.25))

model.add(Conv2D(32, 3, activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(32, 3, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.summary()
model.compile(loss="sparse_categorical_crossentropy",
             optimizer="sgd",
             metrics=['accuracy'])
model.fit(X_train, y_train, epochs=EPOCHS)

## Model Predict
print("Predicting:\n")
prediction = model.predict(X_test)
predictedClass = prediction.argmax(axis=-1)
# Comparing predicted digit to the actual digit value to determine the accuracy of the model
accuracy = np.mean(y_test.ravel() == predictedClass.ravel())
print("Test accuracy:", accuracy)

## Plot miscategorized images
n_rows = 5
n_cols = 10
iWrong = predictedClass != y_test
print(iWrong.sum(), "miscategorized test cases out of", len(y_test))
XWrong = X_test[iWrong]
yWrong = y_test[iWrong]
indices = np.random.choice(len(XWrong), n_rows*n_cols)
plt.figure(figsize=(n_cols * 1.2, n_rows * 1.2))
for row in range(n_rows):
    for col in range(n_cols):
        index = n_cols * row + col
        plt.subplot(n_rows, n_cols, index + 1)
        X = XWrong[[indices[index]]]
        img = X.reshape(imgWidth, imgHeight)
        plt.imshow(img, cmap="binary", interpolation="nearest")
        pred = np.argmax(model.predict(X), axis=-1)[0]
        value = yWrong[indices[index]]
        plt.title("p: {} a: {}".format(pred, value))
        plt.axis('off')
        
plt.subplots_adjust(wspace=0.2, hspace=0.5)
plt.show()
