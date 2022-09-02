import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras import layers
from tensorflow import keras
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import mpld3
from numpy.random import seed as np_seed
from random import seed as python_seed
import base64
import sklearn.metrics as sk_metrics
import seaborn as sn


# turn off tensorflow info messages about e.g. cpu optimization features
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
# no scientific notation
np.set_printoptions(suppress=True, formatter={'float_kind':'{:f}'.format})

def use_cpu_and_make_results_reproducible():
    # Makes GPU invisible to tensorflow
    os.environ['CUDA_VISIBLE_DEVICES'] = ""

    # set python, numpy and tensorflow seeds so that operations
    #   involving randomness can be reperformed consistently
    os.environ['PYTHONHASHSEED'] = "1"
    python_seed(1)

    # numpy seed
    np_seed(1)
    tf.random.set_seed(2)

def denormalize_images(images):
    images = images * 255
    images = images.astype("uint8")
    images = images.reshape((len(images), 28, 28))
    
    return images

def normalize_images(images):
    images = images.reshape((len(images), 28 * 28))
    images = images.astype("float32") / 255

    return images

def prep_data(train_images, test_images):
    train_images = normalize_images(train_images)
    test_images = normalize_images(test_images)

    return train_images,test_images

def value_counts_np(np_array, sort_by_counts=True):
    value_counts = np.unique(np_array, return_counts=True)
    value_counts = pd.Series(data=value_counts[1], index=value_counts[0])
    value_counts.name = 'Value Counts'

    if sort_by_counts:
        value_counts.sort_values(ascending=False, inplace=True)

    return value_counts

def get_history_df(history, pretty_cols=False):

    history_df = pd.DataFrame(history.history)
    history_df.insert(0, 'epoch', range(1, len(history_df) + 1))

    if pretty_cols:
        for col in [x for x in history_df if x != 'epoch']:
            history_df[col] = round(history_df[col] * 100, 3).astype(str).str.ljust(5, '0') + '%'
    
    return history_df

def build_training_log_html(training_log_path):
    log = pd.read_csv(training_log_path, dtype=float)
    html =  f"""
    <html>
      <head>
      </head>
      <body>
        <h2>Training Log</h2>
        <table style="width:100">
            <tr>
              <td>{log.to_html(index=False)}</td>
            </tr>
          </table>
      </body>
    </html>
    """

    return html

def build_fail_counts_html(fail_dist, pred_accuracy):

    # todo move this out of here
    pred_accuracy_pct = str(round(pred_accuracy * 100, 3)) + '%'

    # get value counts sorted by values and by index
    val_counts_1 = pd.DataFrame(fail_dist)
    val_counts_1.index.name = 'Category'
    val_counts_1.reset_index(inplace=True)
    val_counts_2 = val_counts_1.sort_values(by='Category')

    # construct html and write it out
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
          <h2> Prediction Accuracy
          {pred_accuracy_pct}
      </body>
    </html>
    """
    # with open(out_path, 'a') as report:
    #     report.write(html)

    return html

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

def build_training_plot_html(history, plot_type):
    df = get_history_df(history)

    if plot_type == 'accuracy':
        title = 'Training Accuracy'
        y_col_1 = 'accuracy'
        y_col_1_label = 'Training accuracy'
        y_col_2 = 'val_accuracy'
        y_col_2_label = 'Validation accuracy'
    elif plot_type == 'loss':
        title = 'Training Loss'
        y_col_1 = 'loss'
        y_col_1_label = 'Training loss'
        y_col_2 = 'val_loss'
        y_col_2_label = 'Validation loss'
    else:
        raise Exception(f'Invalid plot_type: {plot_type}')
    
    # todo implement side by side chart
    # fig, ax = plt.subplots(1, 2, sharex="col", sharey="row", figsize=(8, 8))
    # fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95,
    #                     hspace=0.1, wspace=0.1)
    
    # instantiate figure
    fig = plt.figure(figsize=(5,5))

    # make y axis a percent
    ax = fig.add_subplot(1,1,1)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))

    # plot
    plt.plot(df['epoch'], df[y_col_1], "bo", label=y_col_1_label)
    plt.plot(df['epoch'], df[y_col_2], "b", label=y_col_2_label)
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel(plot_type)
    plt.legend()

    html = mpld3.fig_to_html(fig)

    return html

def build_fail_images_plot_html(fail_images):
    fail_images = fail_images.copy()
    fail_images = denormalize_images(fail_images)

    plot_rows = 2
    plot_cols = 5

    fig, ax_list = plt.subplots(plot_rows, plot_cols, sharex="col", sharey="row", figsize=(4, 4))
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95,
                        hspace=0.0, wspace=0.0)
    i = 0
    for r in range(plot_rows):
        for c in range(plot_cols):
            ax_list[r][c].imshow(fail_images[i], cmap=plt.cm.binary)
            i += 1

    html = mpld3.fig_to_html(fig)

    return html

def print_processor_type():
    if len(tf.config.list_physical_devices('GPU')) > 0:
        print('Using GPU')
    else:
        print('Using CPU')

def write_model_graph(model, out_path):
    keras.utils.plot_model(model,
                           to_file=out_path,
                           show_shapes=True,
                           show_dtype=True,
                           show_layer_activations=True)

def get_predicted_labels(model, test_data):
    pred = model.predict(test_data)
    pred_labels = []
    for idx, obs in enumerate(pred):
        pred_labels += [np.argmax(obs)]
    return pred_labels

def get_failing_predictions(pred_labels, test_data, test_labels):
    fail_idxs = np.nonzero(pred_labels != test_labels)[0]
    fail_labels = test_labels[fail_idxs]
    fail_data = test_data[fail_idxs]
    fail_dist = value_counts_np(fail_labels)
    return fail_data, fail_dist

def read_model_graph_as_html(model_graph_path):
    data_uri = base64.b64encode(open(model_graph_path, 'rb').read()).decode('utf-8')
    html = f"""
        <br>
        <h2>Model Graph</h2>
        <img src="data:image/png;base64,{data_uri}">
        <br>
    """
    # html_graph = '''<img src="data:image/png;base64,{0}">'''.format(data_uri)
    return html

def build_confusion_matrix_heatmap_html(test_labels, pred_labels):
    conf = sk_metrics.confusion_matrix(test_labels, pred_labels)
    row_col_names = sorted(np.unique(test_labels))
    conf = pd.DataFrame(conf, row_col_names, row_col_names)

    fig = plt.figure(figsize = (7, 7))
    sn.heatmap(conf, annot=True, fmt='d')
    # todo figure out why y axis ticks are cut off in html
    html = mpld3.fig_to_html(fig)

    return html

def write_report(output_paths, model, fail_images, fail_dist, pred_accuracy,
                 history, test_labels, pred_labels):

    report_path = output_paths['report']
    model_graph_path = output_paths['model_graph']
    training_log_path = output_paths['training_log']
    write_model_graph(model, model_graph_path)

    if os.path.exists(report_path):
        os.remove(report_path)
    html_model_graph = read_model_graph_as_html(model_graph_path)
    html_fail_counts = build_fail_counts_html(fail_dist, pred_accuracy)
    html_accuracy = build_training_plot_html(history, 'accuracy')
    html_loss = build_training_plot_html(history, 'loss')
    html_fail_images = build_fail_images_plot_html(fail_images)
    training_log = build_training_log_html(training_log_path)
    confusion_matrix = build_confusion_matrix_heatmap_html(test_labels, pred_labels)

    with open(report_path, 'a') as report:
        report.write(html_accuracy)
        report.write(html_loss)
        report.write(training_log)
        report.write(html_model_graph)
        report.write(html_fail_counts)
        report.write(html_fail_images)
        report.write(confusion_matrix)
        report.close()
        print('done writing report')

usr_path = os.path.expanduser('~/')
output_paths = {'training_log': f'{usr_path}/Desktop/training_log.csv',
                'model_graph': f'{usr_path}/Desktop/model_graph.png',
                'report': f'{usr_path}/Desktop/test2.html'
                }

# todo write data assertions and wrap in a function (e.g. output layer has same # of nodes as # of categories)
# todo plot benchmark line on accuracy graphs
# todo try k-fold cross validation
# todo when needed on other data, figure out the intuition behind fit_transform on training, transform on test

use_cpu_and_make_results_reproducible()
print_processor_type()
(train_data, train_labels), (test_data, test_labels) = mnist.load_data()
train_data, test_data = prep_data(train_data, test_data)

model = keras.Sequential([
    layers.Dense(512, activation="relu"),
    layers.Dense(10, activation="softmax")
])

model.compile(optimizer=keras.optimizers.RMSprop(),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

history = model.fit(train_data,
                    train_labels,
                    shuffle=False,
                    epochs=3,
                    batch_size=128,
                    validation_split=.2,
                    callbacks=keras.callbacks.CSVLogger(output_paths['training_log']))

pred_loss, pred_accuracy = model.evaluate(test_data, test_labels)


# report
pred_labels = get_predicted_labels(model, test_data)
fail_images, fail_dist = get_failing_predictions(pred_labels, test_data, test_labels)
write_report(output_paths, model, fail_images, fail_dist, pred_accuracy, history, test_labels, pred_labels)


