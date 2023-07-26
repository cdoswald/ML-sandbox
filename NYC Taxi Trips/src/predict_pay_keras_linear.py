# -*- coding: utf-8 -*-
"""
Author:         Chris Oswald
Date Created:   23 July 2023
Project:        NYC Taxi Trips
Purpose:        Use Keras embedding layers for categorical features and create
                baseline predictions of driver pay using linear model
"""
# Import packages
import json
import os

import numpy as np
import pandas as pd

import tensorflow as tf
from pyarrow.parquet import read_schema

if __name__ == '__main__':

    # Import config
    with open('../Config/config.json', 'r') as io:
        config = json.load(io)
    dirs, files, params = config['dirs'], config['files'], config['params']

    # Import intermediate data
    intermediate_data_path = os.path.join(
        dirs['intermediate_data'], files['processed_data'],
    )
    schema = read_schema(intermediate_data_path)
    data = pd.read_parquet(intermediate_data_path)
    
    # Drop null observations to avoid error in creating TF dataset
    features_list = [
        'fhv_company',
        'pickup_zone',
        'request_day_of_week',
        'request_hour',
    ]
    data = data.dropna(subset=features_list).reset_index(drop=True)
    
    # Separate features and targets
    features = data.loc[:, features_list].copy()
    targets = data.loc[:, 'driver_pay'].copy()
    assert(len(features) == len(targets))

    # Convert data from Pandas to Tensorflow dataset for modeling
    tf_dataset = tf.data.Dataset.from_tensor_slices((dict(features.dropna()), targets))
    print(tf_dataset.__dict__.keys()) # Inspect attributes

    # Encode categorical variables via embedding layers
    categorical_feature_names = [var for var in features_list]
    
    puzone_input = tf.keras.Input(shape=(), name='puzone', dtype=tf.string)
    puzone_strlookup = tf.keras.layers.StringLookup(
        vocabulary=features['pickup_zone'].unique()
    )
    puzone_strlookup(['Battery Park', 'Jamaica Bay', 'Willets Point'])
    tf.keras.layers.Embedding()
