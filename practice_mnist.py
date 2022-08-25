from tensorflow.keras.datasets import mnist
from tensorflow.keras import layers
from tensorflow import keras
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import mpld3

# turn off tensorflow info messages about e.g. cpu optimization features (turn
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
# no scientific notation
np.set_printoptions(suppress=True, formatter={'float_kind':'{:f}'.format})



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

def get_history_df(history, pretty_cols=False):

    history_df = pd.DataFrame(history.history)
    history_df.insert(0, 'epoch', range(1, len(history_df) + 1))

    if pretty_cols:
        for col in [x for x in history_df if x != 'epoch']:
            history_df[col] = round(history_df[col] * 100, 3).astype(str).str.ljust(5, '0') + '%'
    
    return history_df

def write_output_html(fail_dist, history):

    history_df = get_history_df(history, pretty_cols=True)

    output_path = os.path.expanduser('~/Desktop/test2.html')
    if os.path.exists(output_path):
        os.remove(output_path)

    val_counts_1 = pd.DataFrame(fail_dist)
    val_counts_1.index.name = 'Category'
    val_counts_1.reset_index(inplace=True)
    val_counts_2 = val_counts_1.sort_values(by='Category')
    
    html =  f"""
    <html>
      <head>
      </head>
      <body>
        <h2>Failing Test Observations Distribution</h2>
        <table style="width:50%">
            <tr>
              <td>{val_counts_1.to_html(index=False)}</td>
              <td>{val_counts_2.to_html(index=False)}</td> 
            </tr>
          </table>
          <br>
          <h2>Epoch Summary</h2>
          {history_df.to_html(index=False)}
      </body>
    </html>
    """
    with open(output_path, 'a') as report:
        report.write(html)

    # with open(output_path, 'a') as report:
    #     # todo try building a matplotlib figure that also shows what we want, so we can write it out as html
    #     # t = plt.imshow(fail_images[0].reshape(28,28), cmap=plt.cm.binary)
    #     # mpld3.save_html(t, report)

def shuffle_np(np_arrays, constant_seed=True):
    '''
    np_arrays: np array or tuple of np arrays of the same length
    '''

    # store in a tuple if it's not already
    if type(np_arrays) == np.ndarray:
        np_arrays = (np_arrays,)

    array_len = len(np_arrays[0])

    for array in np_arrays:
        assert len(array) == array_len, 'all arrays must be the same length'

    # keeps the shuffle consistent every re-run
    if constant_seed:
        shuffled_idxs = np.random.RandomState(seed=1).permutation(array_len)
    else:
        shuffled_idxs = np.random.permutation(array_len)

    out = ()
    for np_array in np_arrays:
        out += (np_array[shuffled_idxs],)

    return out

def split_out_validation_obs(train_data, train_labels, validation_frac):
    assert 0 < validation_frac < 1, 'validation frac must be greater than 0 and less than 1'
    assert len(train_data) == len(train_labels), 'training data and labels must be the same length'

    n_validation_obs = int(validation_frac * len(train_data))
    validation_data = train_data[:n_validation_obs]
    train_data = train_data[n_validation_obs:]
    validation_labels = train_labels[:n_validation_obs]
    train_labels = train_labels[n_validation_obs:]

    return train_data, train_labels, validation_data, validation_labels



# Params
validation_frac = .2
#--------

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images, test_images = prep_data(train_images, test_images)
train_images, train_labels, validation_images, validation_labels = \
    split_out_validation_obs(train_images, train_labels, validation_frac)

model = keras.Sequential([
    layers.Dense(512, activation="relu"),
    layers.Dense(10, activation="softmax")
])

model.compile(optimizer="rmsprop",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

history = model.fit(train_images,
                    train_labels,
                    epochs=5,
                    batch_size=128,
                    validation_data=(validation_images, validation_labels))

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

write_output_html(fail_dist, history)

output_path = os.path.expanduser('~/Desktop/test2.html')
df = get_history_df(history)


if os.path.exists(output_path):
    os.remove(output_path)
fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(1,1,1)
ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
plt.plot(df['epoch'], df['accuracy'], "bo", label="Training accuracy")
plt.plot(df['epoch'], df['val_accuracy'], "b", label="Validation accuracy")
plt.plot(df['epoch'], df['loss'], "ro", label="Training loss")
plt.plot(df['epoch'], df['val_loss'], "r", label="Validation loss")
plt.xlabel("Epochs")
plt.ylabel("Accuracy/Loss")
plt.legend()
html = mpld3.fig_to_html(fig)
with open(output_path, 'a') as report:
    report.write(html)
    report.close()






# checkpoint callbacks
# graph training progress
# record spec used


# metadata elements
#   - n training, test, validation datasets
#   - dimensions of an observation (if relevant)
#   - train acc, test acc