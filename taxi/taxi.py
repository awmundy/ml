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
from matplotlib.lines import Line2D

import ml.shared as shared



def write_histogram(df, output_path):
    ignore_cols = ['id']
    histogram_cols = [x for x in df if x not in ignore_cols]

    hist = df[histogram_cols].hist(figsize=(10,10))
    plt.savefig(output_path)

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

def convert_categoricals_to_float(df):
    df['store_and_fwd_flag'] = df['store_and_fwd_flag'].replace({'Y':1.0, 'N': 0.0})
    df['vendor_id'] = df['vendor_id'].astype(float)
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

def remove_0_passenger_count_trips(df):
    msk = np.isclose(df['passenger_count'], 0)
    print(f'removing {msk.sum()} trips where the passenger count was 0')
    df = df[~msk].copy()

    return df

def write_report():
    pass
    # import plotly.express as px
    # fig = px.histogram(train, x='trip_duration')
    #
    # hist_html = fig.to_html()
    # with open(f'{usr_dir}/Documents/ml_taxi/test.html', 'w') as report:
    #     report.write(hist_html)

def remove_outlier_lat_long_trips(df):
    min_lat = 40.5
    max_lat = 41.0
    min_long = -74.05
    max_long = -73.6

    msk = df['pickup_latitude'].between(min_lat, max_lat)
    msk &= df['pickup_longitude'].between(min_long, max_long)
    if 'dropoff_latitude' in df.columns:
        msk &= df['dropoff_latitude'].between(min_lat, max_lat)
        msk &= df['dropoff_longitude'].between(min_long, max_long)
    print(f'removing {(~msk).sum()} trips where the lat/longs were out of range')

    df = df[msk].copy()

    return df

def write_pickup_dropoff_scatterplot_map(train, train_map_path):
    # subset bc the viz gets unreadable due to dropoff points overlapping pickup
    df = train.sample(10000, random_state=1).copy()

    # read in shape file that contains map boundaries
    nyc_geo = gpd.read_file(nyc_boundary_path)

    # construct geopandas dataframes
    pickup_gdf = gpd.GeoDataFrame(df.copy(), geometry=gpd.points_from_xy(df['pickup_longitude'],
                                                                         df['pickup_latitude']))
    dropoff_gdf = gpd.GeoDataFrame(df.copy(), geometry=gpd.points_from_xy(df['dropoff_longitude'],
                                                                          df['dropoff_latitude']))
    # add the boundaries and data to a plot
    ax = nyc_geo.plot(facecolor='whitesmoke', edgecolor='black', figsize=(12,12))
    pickup_gdf.plot(ax=ax, color='purple', marker='^', label='pickup', alpha=.1, markersize=10)
    dropoff_gdf.plot(ax=ax, color='darkorange', marker='v', label='dropoff', alpha=.1, markersize=10)

    # zoom in on the area these points are in
    minx = min(pickup_gdf['pickup_longitude'].min(), dropoff_gdf['dropoff_longitude'].min()) - .01
    maxx = min(pickup_gdf['pickup_longitude'].max(), dropoff_gdf['dropoff_longitude'].max()) +.01
    miny = min(pickup_gdf['pickup_latitude'].min(), dropoff_gdf['dropoff_latitude'].min()) - .01
    maxy = min(pickup_gdf['pickup_latitude'].max(), dropoff_gdf['dropoff_latitude'].max()) + .01
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)

    # make legend
    lines = [Line2D([0], [0], linestyle="none", marker="s", markersize=15,
             markerfacecolor=a.get_facecolor())for a in ax.collections[1:]]
    labels = [t.get_label() for t in ax.collections[1:]]
    ax.legend(lines, labels)

    # write out
    plt.savefig(train_map_path)

usr_dir = os.path.expanduser('~')
train_path = f'{usr_dir}/Documents/ml_taxi/train.csv'
test_path = f'{usr_dir}/Documents/ml_taxi/test.csv'
# https://data.cityofnewyork.us/api/geospatial/tqmj-j8zm?method=export&format=Shapefile
nyc_boundary_path = f'{usr_dir}/Documents/ml_taxi/nyc_borough_geo_files/geo_export_d66f2294-5e4d-4fd3-92f2-cdb0a859ef48.shp'
train_histogram_path = f'{usr_dir}/Documents/ml_taxi/histogram_train.png'
test_histogram_path = f'{usr_dir}/Documents/ml_taxi/histogram_test.png'
train_map_path = f'{usr_dir}/Documents/ml_taxi/map_train.png'
shared.use_cpu_and_make_results_reproducible()
turn_off_scientific_notation()


dtypes, dt_cols = get_dtypes('test')
test = pd.read_csv(test_path, dtype=dtypes, parse_dates=dt_cols)
test = convert_categoricals_to_float(test)
test = remove_0_passenger_count_trips(test)
test = remove_outlier_lat_long_trips(test)
assert test.notnull().all().all()

dtypes, dt_cols = get_dtypes('train')
train = pd.read_csv(train_path, dtype=dtypes, parse_dates=dt_cols)
train = convert_categoricals_to_float(train)
train = remove_0_passenger_count_trips(train)
train = remove_outlier_long_trips(train)
assert train.notnull().all().all()

write_histogram(test, test_histogram_path)
write_histogram(train, train_histogram_path)
write_pickup_dropoff_scatterplot_map(train, train_map_path)
print('done')
