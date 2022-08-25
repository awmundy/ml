from tensorflow.keras.datasets import mnist
from tensorflow.keras import layers
from tensorflow import keras
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mpld3

# turn off tensorflow info messages about e.g. cpu optimization features (turn
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
np.set_printoptions(suppress=True, formatter={'float_kind':'{:f}'.format}) # no scientific notation



def prep_data(train_images, test_images):
    train_images = train_images.reshape((60000, 28 * 28))
    train_images = train_images.astype("float32") / 255
    test_images = test_images.reshape((10000, 28 * 28))
    test_images = test_images.astype("float32") / 255

    return train_images,test_images

def value_counts_np(np_array, sort_by_count=True):
    value_counts = np.unique(np_array, return_counts=True)
    value_counts = pd.Series(data=value_counts[1], index=value_counts[0])
    value_counts.name = 'Value Counts'

    if sort_by_count:
        value_counts.sort_values(ascending=False, inplace=True)

    return value_counts

def write_output_html(fail_dist):
    output_path = os.path.expanduser('~/Desktop/test2.html')
    if os.path.exists(output_path):
        os.remove(output_path)
    with open(output_path, 'a') as report:
        # todo try building a figure that also shows what we want
        # t = plt.imshow(fail_images[0].reshape(28,28), cmap=plt.cm.binary)
        # mpld3.save_html(t, report)

        val_counts_1 = pd.DataFrame(fail_dist)
        val_counts_1.index.name = 'Category'
        val_counts_1.reset_index(inplace=True)
        val_counts_2 = val_counts_1.sort_values(by='Category')
        header = '<b> Incorrect predictions </b> <br><br>'
        report.write(header + val_counts_1.to_html(index=False) +
                     "<br><br>" + val_counts_2.to_html(index=False))

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

# measure accuracy against test data
test_loss, test_acc = model.evaluate(test_images, test_labels)

pred = model.predict(test_images)
max_pred = []
for idx, obs in enumerate(pred):
    max_pred += [np.argmax(obs)]

fail_idxs = np.nonzero(max_pred != test_labels)[0]
fail_labels = test_labels[fail_idxs]
fail_images = test_images[fail_idxs]
fail_dist = value_counts_np(fail_labels)
print(fail_dist)

write_output_html(fail_dist)



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