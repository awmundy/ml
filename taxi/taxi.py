import os
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from statsmodels.stats.outliers_influence import variance_inflation_factor as get_vif
import geopandas as gpd
from matplotlib.lines import Line2D
from sys import platform
from datetime import datetime as dt
import webbrowser

if platform in ('darwin', 'win32'):
    import shared as shared
    import taxi_shared as taxi_shared
else:
    import ml.shared as shared
    import ml.taxi.taxi_shared as taxi_shared

# turn off tensorflow info messages about e.g. cpu optimization features
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

def write_histogram(df, output_path):
    ignore_cols = ['id']
    histogram_cols = [x for x in df if x not in ignore_cols]
    hist = df[histogram_cols].hist(figsize=(20,20))

    plt.savefig(output_path)

def convert_categoricals_to_float(df):
    df['store_and_fwd_flag'] = df['store_and_fwd_flag'].replace({'Y':1.0, 'N': 0.0})
    df['vendor_id'] = df['vendor_id'].astype(float)
    return df

def remove_outlier_long_duration_trips(df):
    trip_cap = 7200
    msk = df['trip_duration'] >= trip_cap
    print(f'removing {msk.sum()} trips lasting {trip_cap/60/60} hours or longer')
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

def remove_outlier_lat_long_trips(df):
    min_lat = 40.5
    max_lat = 41.0
    min_long = -74.05
    max_long = -73.6

    msk = df['pickup_latitude'].between(min_lat, max_lat)
    msk &= df['pickup_longitude'].between(min_long, max_long)
    # todo this isnt necessary as a for loop
    if 'dropoff_latitude' in df.columns:
        msk &= df['dropoff_latitude'].between(min_lat, max_lat)
        msk &= df['dropoff_longitude'].between(min_long, max_long)
    print(f'removing {(~msk).sum()} trips where the lat/longs were out of range')

    df = df[msk].copy()

    return df

def write_pickup_dropoff_scatterplot_map(train, train_map_path):
    # subset bc the viz gets unreadable due to dropoff points overlapping pickup
    df = train.sample(1000, random_state=1).copy()

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

def write_correlation_matrix_heatmap(train, out_path):
    corr_mat = train[sorted(train.columns)].corr()
    corr_mat = corr_mat.round(2)
    fig_dim = float((5 + len(train.columns)))
    fig = plt.figure(figsize = (fig_dim, fig_dim))
    hm = sn.heatmap(corr_mat, annot=True, annot_kws={'size': 15}, cmap='Blues')
    hm.set_xticklabels(hm.get_xmajorticklabels(), fontsize=16, rotation=30)
    hm.set_yticklabels(hm.get_ymajorticklabels(), fontsize=16, rotation=30)
    # make sure laels are on top and bottom
    plt.tick_params(axis='both', which='major',
                    labelbottom = True, bottom=True, top = False, labeltop=True)
    plt.savefig(out_path)

def drop_id_column(df):
    df.drop(columns='id', inplace=True)

    return df

def get_ols_error(train_x, train_y, test_x, test_y):

    # for idx, col in enumerate(train_x.columns):
        # vif = get_vif(train_x, idx)
        # print(col, vif)
        # # todo is a high vif for the constant acceptable
        # if (vif > 10) & (col != 'constant'):
        #     print(f'VIF for {col} is {vif} which is too high')

    model = sm.OLS(train_y, train_x, missing='raise', hasconst=True)
    res = model.fit()
    # print(res.summary2())
    ols_pred = res.predict(test_x)

    # construct mean absolute error
    ols_error = (ols_pred - test_y).abs().sum() / len(ols_pred)
    ols_error = round(ols_error)

    return ols_error

def prep_kaggle_test_data(kaggle_test_path):
    dtypes, dt_cols = taxi_shared.get_dtypes('kaggle_test')
    kaggle_test = pd.read_csv(kaggle_test_path, dtype=dtypes, parse_dates=dt_cols)
    kaggle_test = convert_categoricals_to_float(kaggle_test)
    assert kaggle_test.notnull().all().all()

    return kaggle_test

def get_train_test_val_split(train_original, val_frac, test_frac, y_var):
    assert (val_frac + test_frac) < 1
    start_n = len(train_original)
    train_frac = 1 - val_frac - test_frac
    random_state = 1

    train = train_original.sample(frac=train_frac, random_state=random_state)
    train_original_subset = train_original.drop(train.index)
    
    validation = train_original_subset.sample(frac=val_frac/(val_frac + test_frac), random_state=random_state)
    test = train_original_subset.drop(validation.index)

    assert (len(train) + len(validation) + len(test)) == start_n
    
    train.reset_index(inplace=True, drop=True)
    validation.reset_index(inplace=True, drop=True)
    test.reset_index(inplace=True, drop=True)
    
    train_y = train[y_var].copy()
    train_x = train.drop(columns=y_var)
    validation_y = validation[y_var].copy()
    validation_x = validation.drop(columns=y_var)
    test_y = test[y_var].copy()
    test_x = test.drop(columns=y_var)

    return train_x, train_y, validation_x, validation_y, test_x, test_y

def get_haversine_distance(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points, or all
    pairwise points between two vectors.
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(lambda x: x/360.*(2*np.pi), [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c

    return km

def assign_distance(df, x_vars):
    df['distance'] = get_haversine_distance(df['pickup_longitude'], df['pickup_latitude'],
                                            df['dropoff_longitude'], df['dropoff_latitude'])
    x_vars += ['distance']

    return df, x_vars

def add_time_frequencies(df, x_vars, round_frequency='15min'):
    time_series = df['pickup_datetime'].dt.round(round_frequency).dt.time
    time_series = pd.get_dummies(time_series, prefix='pickup_time', dtype=float)
    vars_to_add = time_series.columns.tolist()
    shape_check = df.shape[0]
    df = df.join(time_series)
    assert shape_check == df.shape[0],\
        f"YOU CLEARLY SHOULD GO BACK TO ACCOUNTING {shape_check == df.shape[0]}. Check your join"
    x_vars += vars_to_add

    return df, x_vars

def add_weekends(df, x_vars):
    weekdays = df['pickup_datetime'].dt.weekday
    weekend_weekday = pd.Series(np.where(weekdays.between(4,6),'weekend', 'weekday'), weekdays.index)
    days_of_week_map = {0:'monday',
                        1:'tuesday',
                        2:'wednesday',
                        3:'thursday',
                        4:'friday',
                        5:'saturday',
                        6:'sunday'}
    day_of_the_week = weekdays.replace(days_of_week_map)
    combo = weekend_weekday.to_frame('day_type').join(day_of_the_week)
    weekday_dummies = pd.get_dummies(combo, dtype=float)
    vars_to_add = weekday_dummies.columns.tolist()
    x_vars += vars_to_add

    shape_check = df.shape[0]
    df = df.join(weekday_dummies)
    assert shape_check == df.shape[0], \
        f"YOU CLEARLY SHOULD GO BACK TO ACCOUNTING {shape_check == df.shape[0]}. Check your join"

    return df, x_vars

def remove_outlier_long_distance_trips(df):
    outlier_distance = 30
    msk = df['distance'] > outlier_distance
    print(f'removing {msk.sum()} trips with distances longer than {outlier_distance}')
    df = df[~msk].copy()

    return df

def write_history_df(history, history_path):
    hist_df = shared.get_history_df(history)
    hist_df.to_csv(history_path, index=False)

def build_val_loss_improvement_compared_to_previous_run_html(history, run_to_compare_against_history_path):
    new_val_loss = history.history['val_loss'][-1]
    old_val_loss = pd.read_csv(run_to_compare_against_history_path)['val_loss'].values[-1]
    improvement = round(old_val_loss - new_val_loss, 3)
    old_run_label = run_to_compare_against_history_path.split('/')[-2]

    html =  f"""
    <html>
      <head>
      </head>
      <body>
        <h3>Improvement in final epoch validation loss compared to run: {old_run_label}</h3>
        <h4>{improvement}</h4>
      </body>
    </html>
    """

    return html

def normalize(df):
    norm_cols = ['passenger_count', 'distance']

    # for each col to be normalized, subtract the mean and divide by the standard deviation to normalize
    mean = df[norm_cols].mean(axis=0)
    df[norm_cols] -= mean
    std = df[norm_cols].std(axis=0)
    df[norm_cols] /= std

    return df

# todo make vif calc more performant and/or hardcode in a minimal set of x vars for ols purposes
# todo implement root mean squared logarithmic error as the error metric (for ols as well?)
# todo add feature: rounded lat long dummies (should improve ols)
# todo add feature: interactions (e.g. borough-time of day)
# todo add feature: airport dummies
# todo query google api to get distance between ~few hundred rounded lat long points,
#  built dataset of road distances between these points
# todo try automated hyperparameter tuning with keras tuner

shared.use_cpu_and_make_results_reproducible()
turn_off_scientific_notation()

# add graphviz to path for those who are blocked from updating it more directly
if "GRAPHVIZ_PATH_EXT" in os.environ.keys():
    os.environ["PATH"] += os.pathsep + os.environ["GRAPHVIZ_PATH_EXT"]

usr_dir = os.path.expanduser('~')
run_time = dt.now().strftime('%Y_%m_%d_%H:%M:%S')
run_dir = f'{usr_dir}/Documents/ml_taxi/runs/{run_time}/'
inputs_dir = f'{usr_dir}/Documents/ml_taxi/'
os.makedirs(run_dir, exist_ok=True)

# input file paths
train_path = f'{inputs_dir}train_w_boro.csv'
kaggle_test_path = f'{inputs_dir}test.csv'
# https://data.cityofnewyork.us/api/geospatial/tqmj-j8zm?method=export&format=Shapefile
nyc_boundary_path = f'{inputs_dir}nyc_borough_geo_files/geo_export_d66f2294-5e4d-4fd3-92f2-cdb0a859ef48.shp'
run_to_compare_against_history_path = f'{inputs_dir}runs/2022_11_08_15:46:42/history.csv'

# output file paths
train_histogram_path = f'{run_dir}histogram_train.png'
test_histogram_path = f'{run_dir}histogram_test.png'
train_map_path = f'{run_dir}map_train.png'
correlation_heatmap_path = f'{run_dir}correlation_heatmap_train.png'
model_graph_path = f'{run_dir}model_graph.png'
model_accuracy_report_path = f'{run_dir}model_accuracy_report.html'
history_path = f'{run_dir}history.csv'

# variables
y_var = 'trip_duration'
# x vars that will definitely be in the model, later vars get optionally added downstream
x_vars = [
    'd_boro_bronx', 'd_boro_brook', 'd_boro_man', 'd_boro_queens', 'd_boro_si',
    'p_boro_bronx', 'p_boro_brook', 'p_boro_man', 'p_boro_queens', 'p_boro_si',
    'passenger_count', 'store_and_fwd_flag', 'vendor_id',
    ]


dtypes, dt_cols = taxi_shared.get_dtypes('train')
train = pd.read_csv(train_path, dtype=dtypes, parse_dates=dt_cols)
train = remove_0_passenger_count_trips(train)
train = remove_outlier_long_duration_trips(train)
train = drop_id_column(train)
train = convert_categoricals_to_float(train)

train, x_vars = assign_distance(train, x_vars)
train = remove_outlier_long_distance_trips(train)
train, x_vars = add_time_frequencies(train, x_vars, '1H')
train, x_vars = add_weekends(train, x_vars)
train = normalize(train)

train = train[x_vars + [y_var]].copy()
assert train.notnull().all().all()

# write out some eda plots
write_histogram(train, train_histogram_path)
# write_pickup_dropoff_scatterplot_map(train, train_map_path)
write_correlation_matrix_heatmap(train, correlation_heatmap_path)

train_x, train_y, validation_x, validation_y, test_x, test_y = get_train_test_val_split(train, .1, .1, y_var)
ols_error = get_ols_error(train_x, train_y, test_x, test_y)

cfg = {'layers': [['relu', 128],
                  ['relu', 128],
                  ['linear', 1]],
       'epochs': 10,
       'batch_size': 10000,
       'learning_rate': .01,
       'loss': 'mae',
       'metrics': ['mean_squared_logarithmic_error'],
       }
cfg['x_vars'] = x_vars

model = keras.Sequential()
for activation, size in cfg['layers']:
    model.add(layers.Dense(size, activation=activation))

model.compile(optimizer=keras.optimizers.RMSprop(cfg['learning_rate']),
              loss=cfg['loss'],
              metrics=cfg['metrics'])

history = model.fit(train_x,
                    train_y,
                    shuffle=False,
                    epochs=cfg['epochs'],
                    batch_size=cfg['batch_size'],
                    validation_data=(validation_x, validation_y)
                    # callbacks=keras.callbacks.CSVLogger('path_to_log_file.txt',
                    # verbose=0
                    )

# miscellaneous writes
write_history_df(history, history_path)
shared.write_model_graph(model, model_graph_path)

# convert plots, etc to html
html_model_graph = shared.read_image_as_html(model_graph_path, 'Model Graph')
html_accuracy = shared.build_training_plot_html(history, cfg['metrics'][0])
html_loss = shared.build_training_plot_html(history, 'loss', ols_error)
html_cfg = shared.convert_dict_to_html(cfg)
html_val_loss_improvement = \
    build_val_loss_improvement_compared_to_previous_run_html(history, run_to_compare_against_history_path)

# write out report
if os.path.exists(model_accuracy_report_path):
    os.remove(model_accuracy_report_path)
with open(model_accuracy_report_path, 'a') as report:
    report.write(html_val_loss_improvement)
    report.write(html_loss)
    report.write(html_accuracy)
    report.write(html_model_graph)
    report.write(html_cfg)
    report.close()
    print('done writing report')
webbrowser.open(model_accuracy_report_path)