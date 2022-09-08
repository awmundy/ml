import os
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras import layers
from tensorflow import keras
import pandas as pd
import requests
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
import ml.shared as shared



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

def one_hot_categoricals(df):
    categorical_cols = ['region']
    for col in categorical_cols:
        dum = pd.get_dummies(df[col], prefix=col)
        dum.drop(dum.columns[-1], axis=1, inplace=True) # drop last one to handle collinearity
        df = pd.concat([df, dum], axis=1)

    return df

def split_to_train_test_and_data_labels(df, test_frac, outcome_var):
    # shuffle, set seed to make reproducible
    df = df.sample(frac=1, random_state=1).reset_index()
    train_data, test_data = train_test_split(df, test_size=test_frac)
    train_labels = train_data.pop(outcome_var)
    test_labels = test_data.pop(outcome_var)

    return train_data, train_labels, test_data, test_labels

def write_report(output_paths, model, pred_accuracy, history):
    #todo write predicted accuracy

    report_path = output_paths['report']
    model_graph_path = output_paths['model_graph']
    training_log_path = output_paths['training_log']
    shared.write_model_graph(model, model_graph_path)

    if os.path.exists(report_path):
        os.remove(report_path)
    html_model_graph = shared.read_image_as_html(model_graph_path, 'Model Graph')
    html_accuracy = shared.build_training_plot_html(history, 'accuracy')
    html_loss = shared.build_training_plot_html(history, 'loss')
    training_log = shared.build_training_log_html(training_log_path)

    with open(report_path, 'a') as report:
        report.write(html_accuracy)
        report.write(html_loss)
        report.write(training_log)
        report.write(html_model_graph)
        report.close()
        print('done writing report')

usr_path = os.path.expanduser('~/')
output_paths = {'training_log': f'{usr_path}/Desktop/housing/training_log.csv',
                'model_graph': f'{usr_path}/Desktop/housing/model_graph.png',
                'report': f'{usr_path}/Desktop/housing/housing_report.html'
                }

# todo if i want to use the vacancy/occupancy data, would have to format it first
# todo run ols as benchmark

download_data()

df = pd.read_excel('/home/amundy/Documents/census_data/housing/2021_mfr_house_puf.xls', dtype=str)
df['constant'] = 1
df.columns = df.columns.str.lower()
df = alter_dtypes(df)
df = one_hot_categoricals(df)


train_data, train_labels, test_data, test_labels = \
    split_to_train_test_and_data_labels(df, .2, 'price')

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

X_cols = ['constant', 'sqft'] + [x for x in df if 'region_' in x]
X = train_data[X_cols].copy()
y = train_labels.copy()

# ols
model = sm.OLS(y, X)
res = model.fit()
res.summary()
ols_pred = res.predict(test_data)

model = keras.Sequential([
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
    ])

model.compile(optimizer=keras.optimizers.RMSprop(),
              loss='mse',
              metrics=['mae'])

history = model.fit(train_data,
                    train_labels,
                    shuffle=False,
                    epochs=5,
                    batch_size=128,
                    validation_split=.2,
                    callbacks=keras.callbacks.CSVLogger(output_paths['training_log'])
                    )

pred_loss, pred_accuracy = model.evaluate(test_data, test_labels)
write_report(output_paths, model, pred_accuracy, history)