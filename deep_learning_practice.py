# until tensorflow fixes a name conflict bug, autocompletion may not work and ide inspection will complain that keras
#   is not available to import
from tensorflow.keras.datasets import mnist
from tensorflow import keras
from tensorflow.keras import layers
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

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

# tensorflow variables are mutable tensors (like np arrays, but have additional useful attributes)
v = tf.Variable(initial_value=tf.ones((3, 1)))
print(v)
v.assign_add(tf.ones((3, 1)))
print(v)

input_const = tf.constant(3.)
with tf.GradientTape() as tape:
   tape.watch(input_const)
   result = tf.square(input_const)
gradient = tape.gradient(result, input_const)
print(result)
print(gradient)

# using gradients to get speed and acceleration
time = tf.Variable(2.)
with tf.GradientTape() as outer_tape:
    with tf.GradientTape() as inner_tape:
        position =  4.9 * time ** 2
    speed = inner_tape.gradient(position, time)
acceleration = outer_tape.gradient(speed, time)
print(position, speed, time)


# building a linear classifier
num_samples_per_class = 1000
negative_samples = np.random.multivariate_normal(
    mean=[0, 3],
    cov=[[1, 0.5],[0.5, 1]],
    size=num_samples_per_class)
positive_samples = np.random.multivariate_normal(
    mean=[3, 0],
    cov=[[1, 0.5],[0.5, 1]],
    size=num_samples_per_class)

inputs = np.vstack((negative_samples, positive_samples)).astype(np.float32)
targets = np.vstack((np.zeros((num_samples_per_class, 1), dtype="float32"),
                     np.ones((num_samples_per_class, 1), dtype="float32")))

# plt.scatter(inputs[:, 0], inputs[:, 1], c=targets[:, 0])
# plt.show()

input_dim = 2
output_dim = 1
W = tf.Variable(initial_value=tf.random.uniform(shape=(input_dim, output_dim)))
b = tf.Variable(initial_value=tf.zeros(shape=(output_dim,)))

def square_loss(targets, predictions):
    per_sample_losses = tf.square(targets - predictions)
    return tf.reduce_mean(per_sample_losses)

def model(inputs, W, b):
    return tf.matmul(inputs, W) + b

def training_step(inputs, targets, W, b, learning_rate):
    with tf.GradientTape() as tape:
        predictions = model(inputs, W, b)
        loss = square_loss(targets, predictions)
    grad_loss_wrt_W, grad_loss_wrt_b = tape.gradient(loss, [W, b])
    W.assign_sub(grad_loss_wrt_W * learning_rate)
    b.assign_sub(grad_loss_wrt_b * learning_rate)
    return loss, W, b

learning_rate = 0.1

# train the model
for step in range(40):
    loss, W, b = training_step(inputs, targets, W, b, learning_rate)
    print(f"Loss at step {step}: {loss:.4f}")

# plot the values the model predicted
predictions = model(inputs, W, b)
plt.scatter(inputs[:, 0], inputs[:, 1], c=predictions[:, 0] > 0.5)
plt.show()