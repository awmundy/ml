import os
import pandas as pd
import requests

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