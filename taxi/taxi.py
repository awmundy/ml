import os
from tensorflow.keras import layers
from tensorflow import keras
import pandas as pd
import requests
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
import numpy as np
import pprint
import matplotlib.pyplot as plt
import seaborn as sn
from statsmodels.stats.outliers_influence import variance_inflation_factor as get_vif
import ml.shared as shared

# no scientific notation
np.set_printoptions(suppress=True, formatter={'float_kind':'{:f}'.format})
pd.set_option('display.float_format', lambda x: '%.6f' % x)

def write_histogram(df, output_path):
    ignore_cols = ['id',
                  'vendor_id',
                  'pickup_longitude',
                  'pickup_latitude',
                  'dropoff_longitude',
                  'dropoff_latitude',
                  ]
    histogram_cols = [x for x in df if x not in ignore_cols]
    n_plots = len(histogram_cols)
    figsize_y = n_plots * 3
    figsize_x = 6

    fig, ax = plt.subplots(nrows=len(histogram_cols), ncols=1, figsize=(figsize_x, figsize_y))
    df.hist(ax=ax, column=histogram_cols)
    fig.savefig(output_path)

def get_dtypes(dataset):
    assert dataset in ['train', 'test']

    dtypes = {'id': 'str',
              'vendor_id': 'str',
              'passenger_count': 'float',
              'pickup_longitude': 'float',
              'pickup_latitude': 'float',
              'dropoff_longitude': 'float',
              'dropoff_latitude': 'float',
              'store_and_fwd_flag': 'str',
              }
    # cols to convert to datetime on read
    dt_cols = ['pickup_datetime']

    if dataset == 'train':
        dtypes['trip_duration'] = 'float'
        dt_cols += ['dropoff_datetime']

    return dtypes, dt_cols

shared.use_cpu_and_make_results_reproducible()