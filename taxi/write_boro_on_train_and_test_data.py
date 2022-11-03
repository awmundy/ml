import fiona
import shapely.geometry
import pandas as pd
import os
import ml.taxi.taxi_shared as taxi_shared
from datetime import datetime as dt
import numpy as np

def assign_pickup_and_dropoff_borough_lambda(row, boro_dict, pickup_cols):
    pickup_assigned = False
    dropoff_assigned = False

    # iterate over boroughs and assign pickup/dropoff borough based on lat/long cols
    for id, boro_info in boro_dict.items():
        boro_bounds = boro_info['bounds']
        pickup_long = row['pickup_longitude']
        pickup_lat = row['pickup_latitude']
        dropoff_long = row['dropoff_longitude']
        dropoff_lat = row['dropoff_latitude']

        if boro_bounds.contains(shapely.geometry.Point(pickup_long, pickup_lat)):
            row['p_boro_' + boro_info['name']] =  1
            pickup_assigned = True
        if boro_bounds.contains(shapely.geometry.Point(dropoff_long, dropoff_lat)):
            row['d_boro_' + boro_info['name']] =  1
            dropoff_assigned = True
        # break out early if possible
        if (pickup_assigned == True) & (dropoff_assigned) == True:
            assert np.isclose(row[pickup_cols].sum(), 1)
            return row

    return row

usr_dir = os.path.expanduser('~')
nyc_boundary_path = f'{usr_dir}/Documents/ml_taxi/nyc_borough_geo_files/geo_export_d66f2294-5e4d-4fd3-92f2-cdb0a859ef48.shp'
train_path = f'{usr_dir}/Documents/ml_taxi/train.csv'
out_path = f'{usr_dir}/Documents/ml_taxi/train_w_boro.csv'

dtypes, dt_cols = taxi_shared.get_dtypes('train')
train = pd.read_csv(train_path, dtype=dtypes, parse_dates=dt_cols)

# create dict to store boro name and the lat long bounds from the shape file
with fiona.open(nyc_boundary_path) as boro_file_object:
    boro_bounds = dict(boro_file_object)
boro_dict = {0: {'name': 'man'},
             1: {'name': 'bronx'},
             2: {'name': 'brook'},
             3: {'name': 'queens'},
             4: {'name': 'si'},
             }
for id in boro_dict.keys():
    boro_dict[id]['bounds'] = shapely.geometry.shape(boro_bounds[id]['geometry'])

# create empty pickup and dropoff cols for each borough
pickup_cols = []
for boro_info in boro_dict.values():
    pickup_col = 'p_boro_' + boro_info['name']
    pickup_cols += [pickup_col]
    dropoff_col = 'd_boro_' + boro_info['name']
    train[pickup_col] = 0
    train[dropoff_col]= 0

# train = train.head(100).copy()
start_time_lambda = dt.now()
train = train.apply(lambda row: assign_pickup_and_dropoff_borough_lambda(row, boro_dict, pickup_cols), axis=1)
end_time_lambda = dt.now()
print(f'runtime of lambda: {(end_time_lambda - start_time_lambda).seconds /60} minutes')

start_time_write = dt.now()
train.to_csv(out_path, index=False)
end_time_write = dt.now()
print(f'runtime of write: {(end_time_write - start_time_write).seconds /60} minutes')
