#!/usr/bin/env python3
##
import numpy as np
import pandas as pd
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
print("tensorflow version", tf.__version__)
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os


## Define image properties:
imgDir = "../../data/images/cats-n-dogs"
Image_Width=128
Image_Height=128
Image_Size=(Image_Width, Image_Height)
Image_Channels=3

## define other constants, including command line argument defaults
batch_size = 30
epochs = 10
modelFName = "cats-n-dogs-model.dat"
train = False

## command line arguments
import __main__ as main
if hasattr(main, "__file__"):
    print("parsing command line arguments")
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", "-d",
                        help = "directory to read images from",
                        default = "images")
    parser.add_argument("--epochs", "-e",
                        help = "how many epochs",
                        default= epochs)
    parser.add_argument("--model", "-m",
                        help = "read/output the model from/to file",
                        default = modelFName)
    parser.add_argument("--train", "-t", action = "store_true",
                        help = "train the model, not just predict")
    args = parser.parse_args()
    imageDir = args.dir
    epochs = int(args.epochs)
    modelFName = args.model
    train = args.train
else:
    print("run interactively from", os.getcwd())
    imageDir = os.path.join(os.path.expanduser("~"),
                            "data", "images", "cats-n-dogs")
print("Load images from", imgDir)
print("epochs:", epochs)
print("model file:", modelFName)
print("train the model:", train)

## Prepare dataset for training model:
if train:
    filenames = os.listdir(os.path.join(imgDir, "train"))
    print(len(filenames), "images found")
    df = pd.DataFrame({
        'filename':filenames,
        'category':pd.Series(filenames).str.slice(0,3)
    })
    print("categories:\n", df.category.value_counts())
    # categories are "dog" and "cat"

    ## Create model
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D,MaxPooling2D,\
         Dropout,Flatten,Dense,Activation,\
         BatchNormalization

    # sequential (not recursive) model (one input, one output)
    model=Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu',
                     input_shape=(Image_Width, Image_Height, Image_Channels)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    model.summary()

    ## Define callbacks and learning rate

    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    earlystop = EarlyStopping(patience = 10)
    learning_rate_reduction = ReduceLROnPlateau(monitor = 'val_accuracy',
                                                # options: loss, accuracy, val_loss, val_accuracy, lr
                                                patience = 2,
                                                verbose = 1,
                                                factor = 0.5,
                                                min_lr = 0.00001)
    callbacks = [earlystop, learning_rate_reduction]

    ## Manage data
    train_df, validate_df = train_test_split(df, test_size=0.20, random_state=42)

    train_df = train_df.reset_index(drop=True)
    validate_df = validate_df.reset_index(drop=True)

    total_train=train_df.shape[0]
    total_validate=validate_df.shape[0]
    print(total_train, "training images and", total_validate, "validation images")

    ## Training and validation data generator:

    train_datagen = ImageDataGenerator(rotation_range=15,
                                    rescale=1./255,
                                    shear_range=0.1,
                                    zoom_range=0.2,
                                    horizontal_flip=True,
                                    width_shift_range=0.1,
                                    height_shift_range=0.1
                                    )

    train_generator = train_datagen.flow_from_dataframe(train_df,
                                                        os.path.join(imgDir, "train"),
                                                        x_col='filename', y_col='category',
                                                        target_size=Image_Size,
                                                        class_mode='categorical',
                                                        batch_size=batch_size)

    validation_datagen = ImageDataGenerator(rescale=1./255)
    validation_generator = validation_datagen.flow_from_dataframe(
        validate_df, 
        os.path.join(imgDir, "train"),
        x_col='filename',
        y_col='category',
        target_size=Image_Size,
        class_mode='categorical',
        batch_size=batch_size
    )

    ## Model Training:
    history = model.fit(
        train_generator, 
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=total_validate//batch_size,
        steps_per_epoch=total_train//batch_size,
        callbacks=callbacks
    )

    ##  Save the model:
    print("Save model to {}".format(modelFName))
    model.save(modelFName)
    # end of training

else:
    ## No training, just load model
    print("Load model from {}".format(modelFName))
    model = tf.keras.models.load_model(modelFName)

## Test data preparation:
testDir = os.path.join(imgDir, "test")
testResults = pd.DataFrame({
    'filename': os.listdir(testDir)
})
nb_samples = testResults.shape[0]
print(nb_samples, "test files read from", testDir)

test_datagen = ImageDataGenerator(
    rescale=1./255
    # do not randomize testing!
)

test_generator = test_datagen.flow_from_dataframe(testResults,
                                                  os.path.join(imgDir, "test"),
                                                  x_col='filename',
                                                  class_mode = None,
                                                  target_size = Image_Size,
                                                  batch_size=batch_size,
                                                  shuffle = False
                                                  # do _not_ randomize the order!
                                                  # this would clash with the file name order!
)
print("test generator done")

## Make categorical prediction:
print("predicting")
phat = model.predict(test_generator, steps=np.ceil(nb_samples/batch_size))
testResults["Pr(cat)"] = phat[:,0]
testResults["Pr(dog)"] = phat[:,1]
print("Predicted array shape:", phat.shape)
print("Example:\n", phat[:5])

## Convert labels to categories:
testResults['category'] = np.argmax(phat, axis=-1)
label_map = {0:"cat", 1:"dog"}
testResults['category'] = testResults['category'].replace(label_map)
rows = np.random.choice(testResults.index, 18, replace=False)
print(testResults.loc[rows])

##  Test your model performance on custom data:
from PIL import Image
import numpy as np
testFiles = ["test/12295.jpg", "test/7089.jpg", "test/740.jpg", "test/3228.jpg", "test/11615.jpg",
             "test/9267.jpg", "test/1492.jpg", "test/2631.jpg",
             "test/1811.jpg", "test/4780.jpg", "test-dog-1.jpg", "test/10804.jpg", "test/11929.jpg", "test/12048.jpg",
             "test/5173.jpg", "test/10542.jpg"]
for testFile in testFiles:
    im = Image.open(os.path.join(imgDir, testFile))
    im=im.resize(Image_Size)
    im=np.expand_dims(im,axis=0)
    im=np.array(im)
    im=im/255
    phat = model.predict([im])
    pred = np.argmax(phat, axis=-1)[0]
    print("sample image {} [{}] is {}".format(testFile, phat, label_map[pred]))

## Visualize the prediction results:
print("Plot example results")
plt.figure(figsize=(12, 24))
index = 1
for row in rows:
    filename = testResults.loc[row, 'filename']
    category = testResults.loc[row, 'category']
    img = load_img(os.path.join(imgDir, "test", filename), target_size=Image_Size)
    plt.subplot(6, 3, index)
    plt.imshow(img)
    plt.xlabel(filename + " ({})".format(category))
    index += 1
plt.tight_layout()
plt.show()
