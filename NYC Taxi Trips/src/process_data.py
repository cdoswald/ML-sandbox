# -*- coding: utf-8 -*-
"""
Author:         Chris Oswald
Date Created:   16 July 2023
Project:        NYC Taxi Trips
Purpose:        Prepare NYC for-hire vehicle (FHV) data for predicting driver pay
                based on pickup location and pickup datetime.

                Time and location are the two main factors that drivers have
                control over before a new trip request comes in (i.e., drivers
                can decide what days/hours to drive and--in some cases--where
                to wait for their next trip to try to maximize per-trip pay)

                Data processing steps:
                - Subset relevant variables
                - Truncate variables (to correct for likely data entry errors)
                - Map FHV license numbers to company names
                - Parse request datetime variable
                - Map pickup location IDs to boroughs/zones
                - Convert trip time from seconds to hours
"""
# Import packages
import json
import os
from timeit import default_timer as timer

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set_theme(style='darkgrid')

from pyarrow.parquet import read_schema

from process_data_functions import truncate_numerical_vars

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
    raw_data_files = sorted([
        file for file in os.listdir(dirs['raw_data'])
        if file.endswith(files['raw_data_suffix'])
    ])
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

    # Map high-volume for-hire vehicle service license number to company name
    data['fhv_company'] = data['hvfhs_license_num'].map({
        'HV0003':'Uber',
        'HV0005':'Lyft',
    })

    # Convert request datetime to weekday (0 = Monday, 6 = Sunday), hour, and minute
    data['request_day_of_week'] = data['request_datetime'].dt.weekday.astype(str)
    data['request_hour'] = data['request_datetime'].dt.hour.astype(str).str.zfill(2)
    data['request_minute'] = data['request_datetime'].dt.minute.astype(str)
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
    data = data.rename(columns={
        'borough':'pickup_borough',
        'zone':'pickup_zone',
    })

    # Convert driver pay from per-trip basis to per-hour basis
    # TODO: account for vehicle operating costs based on trip mileage
    data['trip_time_hours'] = data['trip_time'] / SECONDS_PER_HOUR
    data['implicit_driver_hourly_pay'] = data['driver_pay'] / data['trip_time_hours']

    # Plot average pay by trip request time and location
    groupby_vars_list = [
        'request_day_of_week_and_hour',
        'request_day_of_week',
        'request_hour',
        'pickup_borough',
        'pickup_zone',
    ]
    groupby_results = {}
    for groupby_var in groupby_vars_list:
        groupby_results[groupby_var] = (
            data.groupby([groupby_var])['driver_pay'].mean()
        )
        if len(groupby_results[groupby_var].index) > 20:
            figsize = (12, 32)
        else:
            figsize = (12, 9)
        fig, ax = plt.subplots(figsize=figsize)
        sns.barplot(
            x=groupby_results[groupby_var].values,
            y=groupby_results[groupby_var].index,
            ax=ax,
        )
        ax.set_xlabel('Average Per-Trip Pay')
        figpath = os.path.join(
            dirs['plots'], f'avg_per_trip_pay_by_{groupby_var}.png',
        )
        fig.savefig(figpath, bbox_inches='tight')

    # Export intermediate data
    intermediate_data_path = os.path.join(
        dirs['intermediate_data'], files['processed_data'],
    )
    data.to_parquet(intermediate_data_path, index=False)
