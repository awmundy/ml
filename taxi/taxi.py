import os
# turn off tensorflow info messages about e.g. cpu optimization features
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
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
import matplotlib
# matplotlib.use("Qt5Agg") # backend that a dev requires for plot to work
from sys import platform
from datetime import datetime as dt
import webbrowser
import keras_tuner as kt
from tensorboard import program

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 500)
#pd.set_option('display.width', 1000)

if platform in ('darwin', 'win32'):
    import shared as shared
    import taxi_shared as taxi_shared
else:
    import ml.shared as shared
    import ml.taxi.taxi_shared as taxi_shared

# add graphviz to path for those who are blocked from updating it more directly
if "GRAPHVIZ_PATH_EXT" in os.environ.keys():
    os.environ["PATH"] += os.pathsep + os.environ["GRAPHVIZ_PATH_EXT"]


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
    maxx = min(pickup_gdf['pickup_longitude'].max(), dropoff_gdf['dropoff_longitude'].max()) + .01
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
                    labelbottom=True, bottom=True, top=False, labeltop=True)
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
    ols_mae = (ols_pred - test_y).abs().sum() / len(ols_pred)
    ols_mae = round(ols_mae)

    # root mean squared logarithmic error
    ols_rmsle = (np.sum((np.log(ols_pred + 1) - np.log(test_y))**2)/len(ols_pred))**0.5

    return ols_mae, ols_rmsle

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
    if os.path.exists(run_to_compare_against_history_path):
        new_val_loss = history.history['val_loss'][-1]
        old_val_loss = pd.read_csv(run_to_compare_against_history_path)['val_loss'].values[-1]
        improvement = round(old_val_loss - new_val_loss, 3)
        old_run_label = run_to_compare_against_history_path.split('/')[-2]
        title = f'Improvement in final epoch validation loss compared to run: {old_run_label}'
    else:
        title = 'No valid previous run given to compare against'
        improvement = 'N/A'
    html =  f"""
    <html>
      <head>
      </head>
      <body>
        <h3>{title}</h3>
        <h4>{improvement}</h4>
      </body>
    </html>
    """

    return html

def normalize(df):
    potential_norm_cols = ['passenger_count', 'distance',
                           'pickup_latitude', 'pickup_longitude',
                           'dropoff_latitude', 'dropoff_longitude',]
    norm_cols = []
    for col in potential_norm_cols:
        if col in df:
            norm_cols += [col]
    print(f'normalizing {norm_cols}')

    # for each col to be normalized, subtract the mean and divide by the standard deviation to normalize
    mean = df[norm_cols].mean(axis=0)
    df[norm_cols] -= mean
    std = df[norm_cols].std(axis=0)
    df[norm_cols] /= std

    return df

def add_const(df):
    df.insert(0, 'const', 1)
    return df

def copy_lat_long(df):
    df.insert(0, 'drop_long_raw', df['dropoff_longitude'])
    df.insert(0, 'drop_lat_raw', df['dropoff_latitude'])
    df.insert(0, 'pick_long_raw', df['pickup_longitude'])
    df.insert(0, 'pick_lat_raw', df['pickup_latitude'])
    return df

class CustomHyperModel(kt.HyperModel):
    '''
    Class that is passed to the hyperparameter tuning function
    '''
    def __init__(self, layer_range, node_range, learning_rate_choices,
                 batch_size_choices, loss_choices, metrics):
        self.layer_range = layer_range
        self.node_range = node_range
        self.learning_rate_choices = learning_rate_choices
        self.batch_size_choices = batch_size_choices
        self.loss_choices = loss_choices
        self.metrics = metrics

    # this must be called build or keras won't recognize it
    def build(self, hp):
        '''
        hp: kt.engine.hyperparameters.HyperParameters object
        '''
        model = keras.Sequential()

        # tune number of layers
        for i in range(hp.Int('num_of_layers', self.layer_range[0], self.layer_range[1])):
            # tune layer size
            model.add(keras.layers.Dense(units=hp.Int('#_nodes_l' + str(i),
                                                      min_value=self.node_range[0],
                                                      max_value=self.node_range[1],
                                                      step=32),
                                         activation='relu'))

        model.add(keras.layers.Dense(1, activation='linear'))


        hp_learning_rate = hp.Choice('learning_rate', values=[self.learning_rate_choices[0],
                                                              self.learning_rate_choices[1]])
        hp_loss = hp.Choice('loss_choices', values=[self.loss_choices[0],
                                                    self.loss_choices[1]])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                      loss=hp_loss,
                      metrics=self.metrics)
        return model

    # must be called fit for keras to recognize it, *args/**kwargs are
    # passed in from tuner.search()
    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
                *args,
                batch_size=hp.Choice("batch_size", values=[self.batch_size_choices[0],
                                                           self.batch_size_choices[1]]),
                **kwargs)

def launch_tensorboards(model_fit_log_dir, tuning_log_dir):
    '''
    Launches a tensorboard dashboard for the model fit.
    Also launches a dashboard for hyperparameter tuning if that was done.
    Tensorboard can be launched from the command line directly with:
        tensorboard --logdir='path_to_logs' --port=inert_port_num_here
        (port optional but default port may be already used)
    '''

    tb_model = program.TensorBoard()
    tb_model.configure(argv=[None, '--logdir', model_fit_log_dir])
    url_model = tb_model.launch()
    print(f"Model fit dashboard available on {url_model}")
    if os.path.exists(tuning_log_dir):
        tb_tune = program.TensorBoard()
        tb_tune.configure(argv=[None, '--logdir', tuning_log_dir])
        url_tune = tb_tune.launch()
        print(f"Hyperparameter tuning dashboard available on {url_tune}")

def root_mean_squared_logarithmic_error(y_true, y_pred):
    '''
    Uses tensorflow math functions to return the root mean squared log error as a tensor
    '''
    # used to avoid log(0) issues
    epsilon = tf.constant(0.00001)

    log_error = tf.math.log(y_true + epsilon) - tf.math.log(y_pred + epsilon)
    sq_log_error = tf.pow(log_error, 2)
    mean_sq_log_error = tf.reduce_mean(sq_log_error)
    root_mean_sq_log_error = tf.math.sqrt(mean_sq_log_error)

    return root_mean_sq_log_error

def build_hyperparam_html(best_hps_dict):
    if best_hps_dict:
        html_best_hps_dict = shared.convert_dict_to_html(best_hps_dict)
        header_label = 'Best Hyperparameters from Tuning'
    else:
        html_best_hps_dict = ''
        header_label = 'No Hyperparameter Tuning Performed'
    html_best_hp = \
        f"""
        <html>
          <head>
          </head>
          <body>
            <h2>{header_label}</h2>
            {html_best_hps_dict}
          </body>
        </html>
        """
    return html_best_hp

def build_cfg_html(cfg):
    html_cfg_dict = shared.convert_dict_to_html(cfg)
    html_cfg = \
        f"""
        <html>
          <head>
          </head>
          <body>
            <h2>Run Parameters from cfg</h2>
            {html_cfg_dict}
          </body>
        </html>
        """
    return html_cfg

# todo make vif calc more performant and/or hardcode in a minimal set of x vars for ols purposes
# todo add feature: rounded lat long dummies (should improve ols)
# todo add feature: interactions (e.g. borough-time of day)
# todo add feature: airport dummies
# todo query google api to get distance between ~few hundred rounded lat long points,
#  built dataset of road distances between these points

shared.use_cpu_and_make_results_reproducible()
turn_off_scientific_notation()



# construct run_dir
usr_dir = os.path.expanduser('~')
run_time = dt.now().strftime('%Y_%m_%d_%H:%M:%S')
run_dir = f'{usr_dir}/Documents/ml_taxi/runs/{run_time}/'
os.makedirs(run_dir, exist_ok=True)

# input file paths
inputs_dir = f'{usr_dir}/Documents/ml_taxi/'
train_path = f'{inputs_dir}train_w_boro.csv'
kaggle_test_path = f'{inputs_dir}test.csv'
# https://data.cityofnewyork.us/api/geospatial/tqmj-j8zm?method=export&format=Shapefile
nyc_boundary_path = f'{inputs_dir}nyc_borough_geo_files/geo_export_d66f2294-5e4d-4fd3-92f2-cdb0a859ef48.shp'
run_to_compare_against_history_path = f'{inputs_dir}runs/2022_11_18_09:03:52/history.csv'

# output file paths
model_fit_log_dir = f'{run_dir}logs/'
tuning_dir = f'{run_dir}/hyperparam_tuning/'
tuning_log_dir = f'{tuning_dir}logs/'
train_histogram_path = f'{run_dir}histogram_train.png'
test_histogram_path = f'{run_dir}histogram_test.png'
train_map_path = f'{run_dir}map_train.png'
correlation_heatmap_path = f'{run_dir}correlation_heatmap_train.png'
model_graph_path = f'{run_dir}model_graph.png'
model_accuracy_report_path = f'{run_dir}model_accuracy_report.html'
history_path = f'{run_dir}history.csv'
tuning_dir = f'{run_dir}/hyperparam_tuning/'

# variables
y_var = 'trip_duration'
# x vars that will definitely be in the model, later vars get optionally added downstream
x_vars = [
    # 'd_boro_bronx', 'd_boro_brook', 'd_boro_man', 'd_boro_queens', 'd_boro_si',
    # 'p_boro_bronx', 'p_boro_brook', 'p_boro_man', 'p_boro_queens', 'p_boro_si',
    'const', 'passenger_count', 'store_and_fwd_flag', 'vendor_id',
    'pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude',
    ]

cfg = {'tuning': {'tune?': True, # if true, vals below replace corresponding parts of cfg
                  'layer_range': [1, 2],
                  'node_range': [32, 64], # must be multiples of 32
                  'learning_rate_choices': [.01, .1],
                  'batch_size_choices': [1000, 10000],
                  'loss_choices': ['mae', 'mse']
                  },
       'layers': [['relu', 32],
                  ['relu', 32],
                  # ['relu', 128],
                  # ['relu', 128],
                  # ['relu', 128],
                  ['linear', 1]],
       'epochs': 20,
       'batch_size': 1000,
       'learning_rate': .1,
       'loss': 'mae',
       'metrics': [root_mean_squared_logarithmic_error],
       }

dtypes, dt_cols = taxi_shared.get_dtypes('train')
train = pd.read_csv(train_path, dtype=dtypes, parse_dates=dt_cols,
                    nrows=10000  # todo remove when done testing
                    )
train = add_const(train)
train = remove_0_passenger_count_trips(train)
train = remove_outlier_long_duration_trips(train)
train = drop_id_column(train)
train = convert_categoricals_to_float(train)

# train, x_vars = assign_distance(train, x_vars)
# train = remove_outlier_long_distance_trips(train)
train, x_vars = add_time_frequencies(train, x_vars, '1H')
train, x_vars = add_weekends(train, x_vars)
# todo normalize train/val/test separately
train = normalize(train)

train = train[x_vars + [y_var]].copy()
assert train.notnull().all().all()

train_x, train_y, validation_x, validation_y, test_x, test_y = \
    get_train_test_val_split(train, .1, .1, y_var)
ols_error, ols_rmsle = get_ols_error(train_x, train_y, test_x, test_y)
# print(ols_error, ols_rmsle)

cfg['x_vars'] = x_vars

# hyperparam tuning run that discovers best hyperparameters
if cfg['tuning']['tune?']:
    cfg_t = cfg['tuning']

    # build the hypermodel with hyperparam specified in the config
    tuning_model_builder = \
        CustomHyperModel(layer_range=cfg_t['layer_range'],
                         node_range=cfg_t['node_range'],
                         learning_rate_choices=cfg_t['learning_rate_choices'],
                         batch_size_choices=cfg_t['batch_size_choices'],
                         loss_choices=cfg_t['loss_choices'],
                         metrics=cfg['metrics'])

    # build a tuner object that will be used to search the hyperparameter space
    tuner = kt.Hyperband(tuning_model_builder,
                         # requires the string name of the function
                         objective=kt.Objective('root_mean_squared_logarithmic_error', 'min'),
                         max_epochs=10,
                         directory=tuning_dir,
                         project_name='checkpoints_and_results')

    # iterate through the hyperparam options, discovering the best ones
    tuner.search(train_x, train_y, epochs=10, shuffle=False,
                 validation_data=(validation_x, validation_y),
                 callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=5),
                            keras.callbacks.TensorBoard(tuning_log_dir)])

    # extract best hyperparams from the search (num trials is how many
    # of the best hyperparam sets to return)
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f'Building model with the following tuned values:')
    for k, v in best_hps.values.items():
        print(f'{k}: {v}')
    model = tuner.hypermodel.build(best_hps)
    best_batch_size = best_hps.values['batch_size']

    # train model
    history = model.fit(train_x,
                        train_y,
                        shuffle=False,
                        epochs=cfg['epochs'],
                        batch_size=best_batch_size,
                        validation_data=(validation_x, validation_y),
                        callbacks=[keras.callbacks.TensorBoard(model_fit_log_dir)],
                        )
    model_cfg = model.get_config()
    best_hps_dict = {'num_of_layers': best_hps['num_of_layers'],
                    'learning_rate': best_hps['learning_rate'],
                    'batch_size': best_hps['batch_size'],
                    'loss_metric': best_hps['loss_choices']
                    }
    for param, val in best_hps.values.items():
        if '#_nodes' in param:
            best_hps_dict[param] = val

# non-tuning run that takes params from cfg
else:
    model = keras.Sequential()
    for activation, size in cfg['layers']:
        model.add(layers.Dense(size, activation=activation))

    model.compile(optimizer=keras.optimizers.RMSprop(cfg['learning_rate']),
                  loss=cfg['loss'],
                  metrics=cfg['metrics'])

    # train model
    history = model.fit(train_x,
                        train_y,
                        shuffle=False,
                        epochs=cfg['epochs'],
                        batch_size=cfg['batch_size'],
                        validation_data=(validation_x, validation_y),
                        callbacks=[keras.callbacks.TensorBoard(model_fit_log_dir)],
                        )

# miscellaneous writes
write_histogram(train, train_histogram_path)
# write_pickup_dropoff_scatterplot_map(train, train_map_path)
write_correlation_matrix_heatmap(train, correlation_heatmap_path)
write_history_df(history, history_path)
shared.write_model_graph(model, model_graph_path)

# convert plots, etc to html
html_model_graph = shared.read_image_as_html(model_graph_path, 'Model Graph')
html_accuracy = shared.build_training_plot_html(history, cfg['metrics'][0])
html_loss = shared.build_training_plot_html(history, 'loss', ols_error)
html_cfg = build_cfg_html(cfg)
html_best_hps = build_hyperparam_html(best_hps_dict)

# todo only compare when it's the same loss function
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
    report.write(html_best_hps)
    report.write(html_cfg)
    report.close()
    print('done writing report')
webbrowser.open(model_accuracy_report_path)
launch_tensorboards(model_fit_log_dir, tuning_log_dir)