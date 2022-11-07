def get_dtypes(dataset):
    assert dataset in ['train', 'kaggle_test']

    dtypes = {'id': 'str',
              'vendor_id': 'float',
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