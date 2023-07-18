# -*- coding: utf-8 -*-
"""
Author:         Chris Oswald
Date Created:   16 July 2023
Project:        NYC Taxi Trips
Purpose:        Predict driver pay based on pickup location and pickup datetime.
                These are the two main factors that drivers have control over
                before a new trip request comes in. That is, drivers can decide
                what days/hours to drive and (in some cases) where to idle to
                try to maximize per-trip pay
"""
# Import packages
import json
import os
from timeit import default_timer as timer

import numpy as np
import pandas as pd
import seaborn as sns
sns.set_theme(style='darkgrid')

from pyarrow.parquet import read_schema

from data_processing import truncate_numerical_vars

# Define constants
SECONDS_PER_HOUR = 3600

if __name__ == '__main__':

    # Import config
    with open('../Config/config.json', 'r') as io:
        config = json.load(io)
    dirs, files, params = config['dirs'], config['files'], config['params']

    # Create directories
    for directory in dirs.values():
        os.makedirs(directory, exist_ok=True)

    # Import data
    raw_data_files = [
        file for file in os.listdir(dirs['raw_data'])
        if file.endswith(files['raw_data_suffix'])
    ]
    input_data_path = os.path.join(dirs['raw_data'], raw_data_files[0])
    schema = read_schema(input_data_path)
    var_names = [
        'hvfhs_license_num',
        'dispatching_base_num',
        'originating_base_num',
        'PULocationID',
        'request_datetime',
        'trip_miles',
        'trip_time',
        'driver_pay',
    ]
    data = pd.read_parquet(input_data_path, columns=var_names)

    # Filter variables
    with open('../Config/truncate_vars.json', 'r') as io:
        truncate_vars = json.load(io)
    print(f'Shape of data prior to truncation: {data.shape}')
    data = truncate_numerical_vars(
        df=data,
        truncation_dict=truncate_vars,
        drop_obs=True,
    )
    print(f'Shape of data after truncation: {data.shape}')

    # Convert request datetime to weekday (0 = Monday, 6 = Sunday), hour, and minute
    data['request_day_of_week'] = data['request_datetime'].dt.weekday
    data['request_hour'] = data['request_datetime'].dt.hour
    data['request_minute'] = data['request_datetime'].dt.minute
    data['request_day_of_week_and_hour'] = (
        data['request_day_of_week'].astype(str)
        + '_'
        + data['request_hour'].astype(str)
    )

    # Map pickup location IDs to NYC boroughs, zones, and service zones
    nyc_taxi_zones = pd.read_csv('../Config/nyc_taxi_zone_lookup.csv')
    start_time = timer()
    data = data.merge(
        nyc_taxi_zones,
        how='left',
        left_on='PULocationID',
        right_on='locationID',
        validate='m:1',
    )
    end_time = timer()
    print(f'NYC taxi zone merge took {round(end_time - start_time, 2)} seconds')

    # Convert driver pay from per-trip basis to per-hour basis
    # TODO: account for vehicle operating costs based on trip mileage
    data['trip_time_hours'] = data['trip_time'] / SECONDS_PER_HOUR
    data['driver_hourly_pay'] = data['driver_pay'] / data['trip_time_hours']

    # Plot average driver pay by request time and location
    groupby_vars_list = [
        'request_day_of_week_and_hour',
        'request_day_of_week',
        'request_hour',
        'borough',
        'zone',
    ]
    groupby_results = {}
    for groupby_var in groupby_vars_list:
        groupby_results[groupby_var] = (
            data.groupby([groupby_var])['driver_pay'].mean()
        )
    sns.barplot(
        x=groupby_results['borough'].values,
        y=groupby_results['borough'].index,

    )

    # Linear model

    # K-nearest neighbors

    # Random Forest

    # Gradient boosted decision trees

    # Feedforward neural network

    # Convolutional neural network

    # Recurrent neural network
