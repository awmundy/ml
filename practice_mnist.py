from tensorflow.keras.datasets import mnist
from tensorflow.keras import layers
from tensorflow import keras
import os
import numpy as np

# turn off tensorflow info messages about e.g. cpu optimization features (turn
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
np.set_printoptions(suppress=True, formatter={'float_kind':'{:f}'.format}) # no scientific notation
#


def prep_data(train_images, test_images):
    train_images = train_images.reshape((60000, 28 * 28))
    train_images = train_images.astype("float32") / 255
    test_images = test_images.reshape((10000, 28 * 28))
    test_images = test_images.astype("float32") / 255

    return train_images,test_images

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images, test_images = prep_data(train_images, test_images)

model = keras.Sequential([
    layers.Dense(512, activation="relu"),
    layers.Dense(10, activation="softmax")
])

model.compile(optimizer="rmsprop",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

model.fit(train_images, train_labels, epochs=5, batch_size=128)


# pre process
# model spec
# model compile
# checkpoint callbacks
# graph training progress
# record spec used
# model fit


# metadata elements
#   - n training, test, validation datasets
#   - dimensions of an observation (if relevant)