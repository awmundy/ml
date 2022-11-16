import base64
import mpld3
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
import matplotlib.ticker as ticker
from numpy.random import seed as np_seed
from random import seed as python_seed
import os
import tensorflow as tf
import pprint


def get_history_df(history, pretty_cols=False):

    history_df = pd.DataFrame(history.history)
    history_df.insert(0, 'epoch', range(1, len(history_df) + 1))

    if pretty_cols:
        for col in [x for x in history_df if x != 'epoch']:
            history_df[col] = round(history_df[col] * 100, 3).astype(str).str.ljust(5, '0') + '%'

    return history_df

def read_image_as_html(image_path, image_title=None):
    data_uri = base64.b64encode(open(image_path, 'rb').read()).decode('utf-8')
    html = f"""
        <br>
        <h2>{image_title}</h2>
        <img src="data:image/png;base64,{data_uri}">
        <br>
    """
    return html

def build_training_plot_html(history, plot_type, benchmark=None):
    df = get_history_df(history)

    if plot_type not in df:
        raise Exception(f'{plot_type} not a valid metric in the model history')

    title = f'Training {plot_type}'
    y_col_1 = plot_type
    y_col_1_label = f'Training {plot_type}'
    y_col_2 = f'val_{plot_type}'
    y_col_2_label = f'Validation {plot_type}'

    # todo implement side by side chart
    # fig, ax = plt.subplots(1, 2, sharex="col", sharey="row", figsize=(8, 8))
    # fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95,
    #                     hspace=0.1, wspace=0.1)

    # instantiate figure
    fig = plt.figure(figsize=(5,5))

    # make y axis a percent when relevant
    if plot_type == 'accuracy':
        ax = fig.add_subplot(1,1,1)
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))

    # plot
    if benchmark:
        plt.axhline(y=benchmark, color='r', linestyle='-', label='benchmark'),
    plt.plot(df['epoch'], df[y_col_1], "bo", label=y_col_1_label)
    plt.plot(df['epoch'], df[y_col_2], "b", label=y_col_2_label)
    plt.title(title, fontdict={'fontsize': 16, 'fontweight': 'bold'})
    plt.xlabel("Epochs")
    plt.ylabel(plot_type)
    plt.legend()
    plt.show()

    html = mpld3.fig_to_html(fig)

    plt.close()

    return html

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

def build_test_labels_describe_html(test_labels_describe):
    html =  f"""
    <html>
      <head>
      </head>
      <body>
        <h2>Test Labels Describe</h2>
        <table style="width:100">
            <tr>
              <td>{test_labels_describe.reset_index().to_html(index=False)}</td>
            </tr>
          </table>
      </body>
    </html>
    """

    return html

def write_model_graph(model, out_path):
    keras.utils.plot_model(model,
                           to_file=out_path,
                           show_shapes=True,
                           show_dtype=True)

def use_cpu_and_make_results_reproducible():
    # Makes GPU invisible to tensorflow
    os.environ['CUDA_VISIBLE_DEVICES'] = ""

    # set python, numpy and tensorflow seeds so that operations
    # involving randomness can be reperformed consistently
    os.environ['PYTHONHASHSEED'] = "1"
    python_seed(1)

    # numpy seed
    np_seed(1)
    tf.random.set_seed(2)

def convert_dict_to_html(cfg):
    html_lines = []
    for line in pprint.pformat(cfg, sort_dicts=False).splitlines():
        html_lines.append(f'<br/>{line}')
    html = '\n'.join(html_lines)

    return html