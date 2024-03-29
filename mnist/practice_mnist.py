import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras import layers
from tensorflow import keras
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mpld3
import sklearn.metrics as sk_metrics
import seaborn as sn
from keras import backend as keras_backend
import ml.shared as shared

# turn off tensorflow info messages about e.g. cpu optimization features
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
# no scientific notation
np.set_printoptions(suppress=True, formatter={'float_kind':'{:f}'.format})
pd.set_option('display.float_format', lambda x: '%.6f' % x)

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

def write_confusion_matrix_heatmap(test_labels, pred_labels, out_path):
    conf = sk_metrics.confusion_matrix(test_labels, pred_labels)
    row_col_names = sorted(np.unique(test_labels))
    conf = pd.DataFrame(conf, row_col_names, row_col_names)

    fig = plt.figure(figsize = (7, 7))
    sn.heatmap(conf, annot=True, fmt='d')
    plt.savefig(out_path)


def get_predicted_labels(model, test_data, collapse_pred=True):
    pred = model.predict(test_data)
    pred_labels = []
    for idx, obs in enumerate(pred):
        if collapse_pred:
            pred_labels += [np.argmax(obs)]
        else:
            pred_labels += [obs]
    return pred_labels

def get_failing_predictions(pred_labels, test_data, test_labels):
    fail_idxs = np.nonzero(pred_labels != test_labels)[0]
    fail_labels = test_labels[fail_idxs]
    fail_data = test_data[fail_idxs]
    fail_dist = value_counts_np(fail_labels)
    return fail_data, fail_dist

def write_report(output_paths, model, fail_images, fail_dist, pred_accuracy,
                 history, test_labels, pred_labels):

    report_path = output_paths['report']
    model_graph_path = output_paths['model_graph']
    confusion_heatmap_path = output_paths['confusion_heatmap']
    training_log_path = output_paths['training_log']
    shared.write_model_graph(model, model_graph_path)
    write_confusion_matrix_heatmap(test_labels, pred_labels, confusion_heatmap_path)

    if os.path.exists(report_path):
        os.remove(report_path)
    html_model_graph = shared.read_image_as_html(model_graph_path, 'Model Graph')
    html_fail_counts = build_fail_counts_html(fail_dist, pred_accuracy)
    html_accuracy = shared.build_training_plot_html(history, 'accuracy')
    html_loss = shared.build_training_plot_html(history, 'loss')
    html_fail_images = build_fail_images_plot_html(fail_images)
    training_log = shared.build_training_log_html(training_log_path)
    confusion_heatmap = shared.read_image_as_html(confusion_heatmap_path, 'Confusion Matrix')

    with open(report_path, 'a') as report:
        report.write(html_accuracy)
        report.write(html_loss)
        report.write(training_log)
        report.write(html_model_graph)
        report.write(html_fail_counts)
        report.write(html_fail_images)
        report.write(confusion_heatmap)
        report.close()
        print('done writing report')

def get_node_activations_and_params(model, model_input_data):
    '''
    params:
        model: keras model object
        model_input_data: the data to feed into the model, e.g. test data, training data, etc

    Collects node activations, weights, and biases for each layer after the input layer.
    Node activations for a hidden layer are not stored anywhere in the model. To derive them,
    the data from the previous layer need to be passed to the layer of interest, and
    the node activations then recorded. Weights and biases can be extracted without
    passing data to the layer.

    returns: dictionary where the keys are the layer number and the activations are dictionaries
             containing 3 dataframes for the activations, weights, and biases respectively, i.e.
             node_metrics = {'l0': {'activations': df, 'weights': df, 'biases': df},
                             'l1': {...},
                              ... }

    '''

    # output container
    node_metrics = {}

    # for the first round, the layer_input_data is just the data fed into the model
    layer_input_data = model_input_data

    # The input layer does not count as a layer, so the first hidden layer is layer 0
    for layer_number in range(len(model.layers)):
        layer_label = 'l' + str(layer_number)
        node_metrics[layer_label] = {}

        # these are tensors representing the structure of the input and output
        # data for the layer
        input_tensor = model.layers[layer_number].input
        output_tensor = model.layers[layer_number].output

        # returns a function that takes input data for the layer and returns an
        # np array of the node activations
        get_layer_output = keras_backend.function([input_tensor], [output_tensor])

        # get the node activations given the input from the previous layer
        output = get_layer_output(layer_input_data)
        # # unwrap from outer list for easier inspection
        output = output[0]

        # construct df of node activations
        node_labels = ['node_' + str(x) for x in range(len(output[0]))]
        act_df = pd.DataFrame(columns=node_labels, data=output)
        act_df.insert(0, 'obs', range(len(act_df)))
        act_df['layer'] = layer_number
        node_metrics[layer_label]['activations'] = act_df

        # retrieve the weights and biases for the layer
        params = model.layers[layer_number].get_weights()

        # For a dense layer, there is one weight array per feature in the layer before it. Each weight
        # array has one weight per node in the layer.
        # Ex: For MNIST there are 784 features (pixels), so there are 784 weight arrays in the
        # first hidden layer.
        # If this hidden layer has 20 nodes, each weight array will be length 20.
        # The total number of weights is therefore:
        #       (# of features in the preceding layer * number of nodes in the layer)
        weights = params[0]
        weight_df = pd.DataFrame(columns=node_labels, data=weights)
        weight_df.insert(0, 'previous_layer_node', range(len(weight_df)))
        weight_df['layer'] = layer_number
        node_metrics[layer_label]['weights'] = weight_df

        # 1 bias value per node in the layer, so biases length == number of nodes in the layer
        biases = params[1]
        bias_df = pd.DataFrame(columns=node_labels, data=[biases])
        bias_df['layer'] = layer_number
        node_metrics[layer_label]['biases'] = bias_df

        # reset the layer_input_data for next round
        layer_input_data = output

    # QA
    # get the predicted labels (as probabilities) using model.predict
    pred_label_probabilities = get_predicted_labels(model, model_input_data, collapse_pred=False)
    pred_df = pd.DataFrame(columns=node_labels, data=pred_label_probabilities)

    # qa the output layer produced from this function against the output from model.predict
    output_layer_number = len(model.layers) - 1
    qa_df = node_metrics[f'l{output_layer_number}']['activations'].copy()
    qa_df = qa_df[[x for x in pred_df]]
    pd.testing.assert_frame_equal(pred_df, qa_df)

    return node_metrics

# todo write data assertions and wrap in a function (e.g. output layer has same # of nodes as # of categories)
# todo plot benchmark line on accuracy graphs
# todo try k-fold cross validation
# todo when needed on other data, figure out the intuition behind fit_transform on training, transform on test

usr_path = os.path.expanduser('~/')
output_paths = {'training_log': f'{usr_path}/Desktop/training_log.csv',
                'model_graph': f'{usr_path}/Desktop/model_graph.png',
                'confusion_heatmap': f'{usr_path}/Desktop/confusion_heatmap.png',
                'report': f'{usr_path}/Desktop/test2.html'
                }

shared.use_cpu_and_make_results_reproducible()
print_processor_type()
(train_data, train_labels), (test_data, test_labels) = mnist.load_data()
train_data, test_data = prep_data(train_data, test_data)

model = keras.Sequential([
    layers.Dense(20, activation="relu"),
    layers.Dense(15, activation="softmax")
])

model.compile(optimizer=keras.optimizers.RMSprop(),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

history = model.fit(train_data,
                    train_labels,
                    shuffle=False,
                    epochs=1,
                    batch_size=128,
                    validation_split=.2,
                    callbacks=keras.callbacks.CSVLogger(output_paths['training_log']))

pred_loss, pred_accuracy = model.evaluate(test_data, test_labels)


# report
pred_labels = get_predicted_labels(model, test_data)
fail_images, fail_dist = get_failing_predictions(pred_labels, test_data, test_labels)
write_report(output_paths, model, fail_images, fail_dist, pred_accuracy, history, test_labels, pred_labels)

node_metrics = get_node_activations_and_params(model, test_data)
