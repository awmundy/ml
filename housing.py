import os
import pandas as pd
import requests
import statsmodels.api as sm


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

# todo if i want to use the vacancy/occupancy data, would have to format it first
# todo run ols as benchmark

download_data()

df = pd.read_excel('/home/amundy/Documents/census_data/housing/2021_mfr_house_puf.xls', dtype=str)
df.columns = df.columns.str.lower()
df = alter_dtypes(df)
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




categorical_cols = ['region']
for col in categorical_cols:
    dum = pd.get_dummies(df[col], prefix=col)
    dum.drop(dum.columns[-1], axis=1, inplace=True) # drop last one to handle collinearity
    df = pd.concat([df, dum], axis=1)

df['constant'] = 1

X_cols = ['constant', 'sqft'] + [x for x in df if 'region_' in x]
# X_cols = ['constant', 'sqft']
X = df[X_cols].copy()
y = df['price'].copy()

# ols
model = sm.OLS(y, X)
res = model.fit()
res.summary()