import os
from tensorflow.keras import layers
from tensorflow import keras
import pandas as pd
import statsmodels.api as sm
import numpy as np
import pprint
import matplotlib.pyplot as plt
import seaborn as sn
from statsmodels.stats.outliers_influence import variance_inflation_factor as get_vif
import geopandas as gpd
import ml.shared as shared



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



def convert_store_and_fwd_flag_to_float(df):
    df['store_and_fwd_flag'] = df['store_and_fwd_flag'].replace({'Y':1.0, 'N': 0.0})

    return df

def remove_outlier_long_trips(df):
    trip_cap = 7200
    msk = df['trip_duration'] >= trip_cap
    print(f'removing {msk.sum()} trips lasting {7200/60/60} hours or longer')
    df = df[~msk].copy()

    return df

def turn_off_scientific_notation():
    #numpy
    np.set_printoptions(suppress=True, formatter={'float_kind':'{:f}'.format})
    # pandas
    pd.set_option('display.float_format', lambda x: '%.6f' % x)
    # matplotlib axes
    plt.rcParams["axes.formatter.limits"] = (-5, 12)

def write_report():
    pass
    # import plotly.express as px
    # fig = px.histogram(train, x='trip_duration')
    #
    # hist_html = fig.to_html()
    # with open(f'{usr_dir}/Documents/ml_taxi/test.html', 'w') as report:
    #     report.write(hist_html)

usr_dir = os.path.expanduser('~')
train_path = f'{usr_dir}/Documents/ml_taxi/train.csv'
test_path = f'{usr_dir}/Documents/ml_taxi/test.csv'
train_histogram_path = f'{usr_dir}/Documents/ml_taxi/histogram_train.png'
test_histogram_path = f'{usr_dir}/Documents/ml_taxi/histogram_test.png'
shared.use_cpu_and_make_results_reproducible()
turn_off_scientific_notation()



dtypes, dt_cols = get_dtypes('test')
test = pd.read_csv(test_path, dtype=dtypes, parse_dates=dt_cols)
# convert Y/N col to float
test = convert_store_and_fwd_flag_to_float(test)
assert test.notnull().all().all()
dtypes, dt_cols = get_dtypes('train')
train = pd.read_csv(train_path, dtype=dtypes, parse_dates=dt_cols)
train = convert_store_and_fwd_flag_to_float(train)
train = remove_outlier_long_trips(train)


write_histogram(test, test_histogram_path)
write_histogram(train, train_histogram_path)


