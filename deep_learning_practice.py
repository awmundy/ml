# until tensorflow fixes a name conflict bug, autocompletion may not work and ide inspection will complain that keras
#   is not available to import
from tensorflow.keras.datasets import mnist
from tensorflow import keras
from tensorflow.keras import layers
import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # turn off tensorflow info messages about e.g. cpu optimization features
np.set_printoptions(suppress=True, formatter={'float_kind':'{:f}'.format}) # no scientific notation

# load test and training data- images are np arrays of pixel darkness
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# construct a keras model with two layers
model = keras.Sequential([
    layers.Dense(512, activation="relu"),
    layers.Dense(10, activation="softmax")
])

# configure the model
model.compile(optimizer="rmsprop", # how the model improves itself
              loss="sparse_categorical_crossentropy", # type of loss function used by model to measure its accuracy
              metrics=["accuracy"]) # metric the model evaluates, typically accuracy, could be false negatives etc.

# preprocessing
# - reshape data to be 60000 arrays of length 784, with array values as floats between 0 and 1
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32") / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype("float32") / 255

# train the model
model.fit(train_images, train_labels, epochs=5, batch_size=128)

# test the model on some of the test data
test_digits = test_images[0:10]
predictions = model.predict(test_digits)
print('probabilities for each number 0-9 that the test digit is that number')
print(predictions[0])

# get accuracy of model on the test data
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"test_acc: {test_acc}")