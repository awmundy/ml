import os
import tensorflow as tf
from tensorflow.keras.datasets import mnist
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
import mpld3
from statsmodels.stats.outliers_influence import variance_inflation_factor as get_vif

import ml.shared as shared


# turn off tensorflow info messages about e.g. cpu optimization features
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
# no scientific notation
np.set_printoptions(suppress=True, formatter={'float_kind':'{:f}'.format})
pd.set_option('display.float_format', lambda x: '%.6f' % x)


def download_data():

    urls = {'2017_nyc_occupied.txt':
                'https://www2.census.gov/programs-surveys/nychvs/datasets/2017/microdata/uf_17_occ_web_b.txt',
            '2017_nyc_occupied_layout.pdf':
                'https://www2.census.gov/programs-surveys/nychvs/technical-documentation/record-layouts/2017/occupied-units-17.pdf',
            '2017_nyc_vacant.txt':
                'https://www2.census.gov/programs-surveys/nychvs/datasets/2017/microdata/uf_17_vac_web_b.txt',
            '2017_nyc_vacant_layout.pdf':
                'https://www2.census.gov/programs-surveys/nychvs/technical-documentation/record-layouts/2017/vacant-units-17.pdf',
            '2021_mfr_house_puf.xls':
                'https://www2.census.gov/programs-surveys/mhs/tables/2021/PUF2021final_v1.xls',
            '2021_mfr_house_puf_layout.pdf':
                'https://www.census.gov/content/dam/Census/programs-surveys/mhs/technical-documentation/puf/MHS_PUF_Documentation2021.pdf'
            }

    out_dir = '/home/amundy/Documents/census_data/housing/'
    os.makedirs(out_dir, exist_ok=True)

    for file_name, url in urls.items():
        out_path = f'{out_dir}{file_name}'
        # don't redownload
        if os.path.exists(out_path):
            continue

        res = requests.get(url)
        if not res.ok:
            raise Exception(f'The following zip file url failed to download: {url}')

        with open(out_path, 'wb') as output_location:
            output_location.write(res.content)
            print(f'done downloading {out_path}')

def alter_dtypes(df):
    float_cols = ['price', 'sqft']
    for col in float_cols:
        df[col] = df[col].astype(float)

    return df

def one_hot_categoricals(df, categorical_cols):
    if len(categorical_cols) > 0:
        for col in categorical_cols:
            dum = pd.get_dummies(df[col], prefix=col)
            dum.drop(dum.columns[-1], axis=1, inplace=True) # drop last one to handle collinearity
            df = pd.concat([df, dum], axis=1)

    return df

def split_to_train_test_and_data_labels(df, test_frac, outcome_var):
    train_data, test_data = train_test_split(df, test_size=test_frac, random_state=1)
    train_data.reset_index(drop=True, inplace=True)
    test_data.reset_index(drop=True, inplace=True)
    train_labels = train_data.pop(outcome_var)
    test_labels = test_data.pop(outcome_var)

    return train_data, train_labels, test_data, test_labels

def build_prediction_error_html(pred_error, ols_error):
    html =  f"""
    <html>
      <head>
      </head>
      <body>
        <h2>Neural Net Prediction Error per Obs</h2>
        {pred_error}
        <h2>OLS Prediction Error per Obs</h2>
        {ols_error}
      </body>
    </html>
    """

    return html

def convert_dict_to_html(cfg):
    html_lines = []
    for line in pprint.pformat(cfg, sort_dicts=False).splitlines():
        html_lines.append(f'<br/>{line}')
    html = '\n'.join(html_lines)

    return html

def write_report(model, history, ols_error, test_labels, test_data, cfg):

    metric = cfg['metrics']
    assert len(metric) == 1
    metric = metric[0]

    report_path = cfg['out_paths']['report']
    model_graph_path = cfg['out_paths']['model_graph']
    training_log_path = cfg['out_paths']['training_log']
    corr_heatmap_path = cfg['out_paths']['corr_heatmap']
    histogram_path = cfg['out_paths']['histogram']

    shared.write_model_graph(model, model_graph_path)

    # get per observation prediction error
    # _, p = model.evaluate(test_data, test_labels)
    pred = model.predict(test_data)
    pred_error = (pred.flatten() - test_labels).abs().sum() / len(test_labels)
    pred_error = round(pred_error)

    test_labels_describe = test_labels.describe()

    if os.path.exists(report_path):
        os.remove(report_path)
    html_model_graph = shared.read_image_as_html(model_graph_path, 'Model Graph')
    html_accuracy = shared.build_training_plot_html(history, metric)
    html_loss = shared.build_training_plot_html(history, 'loss')
    html_training_log = shared.build_training_log_html(training_log_path)
    html_pred_error = build_prediction_error_html(pred_error, ols_error)
    html_cfg = convert_dict_to_html(cfg)
    html_corr_heatmap = shared.read_image_as_html(corr_heatmap_path, 'Correlation Matrix')
    html_histogram = shared.read_image_as_html(histogram_path, 'Features Histogram')
    html_test_labels_describe = shared.build_test_labels_describe_html(test_labels_describe)

    with open(report_path, 'a') as report:
        report.write(html_cfg)
        report.write(html_pred_error)
        report.write(html_test_labels_describe)
        report.write(html_accuracy)
        report.write(html_loss)
        # report.write(html_training_log)
        report.write(html_histogram)
        report.write(html_corr_heatmap)
        report.write(html_model_graph)
        report.close()
        print('done writing report')

def normalize(df, cols):
    mean = df[cols].mean(axis=0)
    df[cols] -= mean
    std = df[cols].std(axis=0)
    df[cols] /= std

    return df

def get_ols_error(train_data, train_labels, test_data):
    for idx, col in enumerate(train_data.columns):
        vif = get_vif(train_data, idx)
        # todo is a high vif for the constant acceptable
        if (vif > 10) & (col != 'constant'):
            print(f' VIF for {col} is {vif} which is ')


    model = sm.OLS(train_labels, train_data, missing='raise', hasconst=True)
    res = model.fit()
    # print(res.summary2())
    ols_pred = res.predict(test_data)

    # construct mean absolute error
    ols_error = (ols_pred - test_labels).abs().sum() / len(ols_pred)
    ols_error = round(ols_error)

    return ols_error

def write_correlation_matrix_heatmap(train_data, train_labels, out_path):
    df = pd.concat([train_data, train_labels], axis=1).drop(columns='constant')
    corr_mat = df.corr()
    corr_mat = corr_mat.round(2)
    fig_dim = float((3 + len(train_data.columns)))
    fig = plt.figure(figsize = (fig_dim, fig_dim))
    sn.heatmap(corr_mat, annot=True)
    plt.savefig(out_path)

def write_histogram_for_raw_data_numeric_cols(df, output_path):
    num_cols = ['status', 'finaldest', 'footings', 'lease', 'region', 'piers',
                'secured', 'titled', 'sections', 'price', 'sqft', 'bedrooms',
                'location']
    df = df.copy()
    for col in num_cols:
        df[col] = df[col].astype(float)
    hist = df[num_cols].hist(figsize=(10,10))
    plt.savefig(output_path)

def get_feature_list(df, non_categorical_cols, categorical_cols):
    features = non_categorical_cols
    for col in categorical_cols:
        features += [x for x in df if col + '_' in x]

    return features

def drop_records_with_no_price(df):
    msk = np.isclose(df['price'], 9)
    df = df[~msk].copy()

    return df




# todo if i want to use the vacancy/occupancy data, would have to format it first
# todo why does normalization cause convergence to take longer?


usr_path = os.path.expanduser('~/')
cfg = {'out_paths':{'training_log': f'{usr_path}/Desktop/housing/training_log.csv',
                    'model_graph': f'{usr_path}/Desktop/housing/model_graph.png',
                    'corr_heatmap': f'{usr_path}/Desktop/housing/corr_heatmap.png',
                    'histogram': f'{usr_path}/Desktop/housing/histogram.png',
                    'report': f'{usr_path}/Desktop/housing/housing_report.html'
                    },
       'layers': [['relu', 64],
                  ['relu', 64],
                  ['linear', 1]],
       'epochs': 50,
       'batch_size': 32,
       'loss': 'mae',
       'metrics': ['mae'],
       'normalized_features': ['sqft'],
       }

shared.use_cpu_and_make_results_reproducible()
download_data()
df = pd.read_excel('/home/amundy/Documents/census_data/housing/2021_mfr_house_puf.xls', dtype=str)
df['constant'] = 1
df.columns = df.columns.str.lower()
df = alter_dtypes(df)
df = drop_records_with_no_price(df)
write_histogram_for_raw_data_numeric_cols(df, cfg['out_paths']['histogram'])
categorical_cols = \
    [
        'region',
        'sections',
        'finaldest',
        'footings',
        'secured',
        'bedrooms'
        ]
non_categorical_cols = \
    [
        'constant',
        'sqft'
        ]

df = one_hot_categoricals(df, categorical_cols)
features = get_feature_list(df, non_categorical_cols, categorical_cols)
cfg['features'] = features

train_data, train_labels, test_data, test_labels = \
    split_to_train_test_and_data_labels(df, .2, 'price')


train_data = train_data[features].copy()
test_data = test_data[features].copy()
ols_error = get_ols_error(train_data, train_labels, test_data)
write_correlation_matrix_heatmap(train_data, train_labels, cfg['out_paths']['corr_heatmap'])

if 'normalized_features' in cfg.keys():
    train_data = normalize(train_data, cfg['normalized_features'])
    test_data = normalize(test_data, cfg['normalized_features'])

model = keras.Sequential()
for activation, size in cfg['layers']:
    model.add(layers.Dense(size, activation=activation))

model.compile(optimizer=keras.optimizers.RMSprop(),
              loss=cfg['loss'],
              metrics=cfg['metrics'])

history = model.fit(train_data,
                    train_labels,
                    shuffle=False,
                    epochs=cfg['epochs'],
                    batch_size=cfg['batch_size'],
                    validation_split=.2,
                    callbacks=keras.callbacks.CSVLogger(cfg['out_paths']['training_log']),
                    # verbose=0
                    )

write_report(model, history, ols_error, test_labels, test_data, cfg)

### NOTES ###
# explanatory variables
# region: Census region, "USA" region (5) for a small subset of homes, [1, 2, 3, 4, 5]
# section: Size of home, [1, 2, 3]
# finaldest: Whether home is placed at final destination, [1, 2]
# sqft: Square footage, with winsorization [rounded winsorized sqft]
# bedrooms: Number of bedrooms, 2 or less, 3 or more, na or privacy  [1, 2, 9]
# location: Inside manuf. home community, outside, na or privacy [1, 3, 9]
# footings: Type of footings, [1, 2, 3, 4, 5, 9]
# piers: Type of piers, partially dependent on footings, [0, 1, 2, 3, 4, 9]
# secured: How it is secured, partially dependent on footings, [0, 1, 2, 3, 9]
# lots of the data is imputed

