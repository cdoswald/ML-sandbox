# -*- coding: utf-8 -*-
"""
Author:         Chris Oswald
Date Created:   23 July 2023
Project:        NYC Taxi Trips
Purpose:        Test Keras embedding layers for categorical features
"""
# Import packages
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
    
import tensorflow as tf
from pyarrow.parquet import read_schema

from plotting_functions import plot_train_and_val_loss

if __name__ == '__main__':

    # Import config
    with open('../config/config.json', 'r') as io:
        config = json.load(io)
    dirs, files, params = config['dirs'], config['files'], config['params']

    # Import intermediate data
    intermediate_data_path = os.path.join(
        dirs['intermediate_data'], files['processed_data'],
    )
    schema = read_schema(intermediate_data_path)
    data = pd.read_parquet(intermediate_data_path).iloc[:50000]
    
    # Drop null observations to avoid error in creating TF dataset
    categorical_features = [
        'fhv_company',
        'pickup_zone',
        'request_day_of_week',
        'request_hour',
    ]
    data = data.dropna(subset=categorical_features).reset_index(drop=True)
    
    # Separate features and targets
    features = data.loc[:, categorical_features].copy()
    targets = data.loc[:, 'driver_pay'].copy()
    assert(len(features) == len(targets))

    # # Convert data from Pandas to Tensorflow dataset for modeling
    # tf_dataset = tf.data.Dataset.from_tensor_slices(
    #     (dict(features.dropna()), targets)
    # )
    # print(tf_dataset.__dict__.keys()) # Inspect attributes

    # Encode categorical variables via embedding layers
    input_layers = []
    embedding_layers = []
    for feature in categorical_features:
        input_layer = tf.keras.Input(
            shape=(1,),
            name=f'input_{feature}',
            dtype=tf.string,
        )
        # input_layer = input_layer[:, tf.newaxis]
        input_layers.append(input_layer)
        str_lookup_layer = tf.keras.layers.StringLookup(
            vocabulary=features[feature].unique(),
            output_mode='int',
            name=f'str_lookup_{feature}',
        )(input_layer)
        embedding_layer = tf.keras.layers.Embedding(
            input_dim=features[feature].nunique() + 1,
            output_dim=4,
            input_length=1,
            trainable=True,
            name=f'embed_layer_{feature}',
        )(str_lookup_layer)
        embedding_layers.append(tf.keras.layers.Flatten()(embedding_layer))
    concat_embed_layer = tf.keras.layers.Concatenate()(embedding_layers)
    dense_layer = tf.keras.layers.Dense(1, activation='relu')(concat_embed_layer)
    model = tf.keras.Model(inputs=input_layers, outputs=dense_layer)        
    tf.keras.utils.plot_model(model, show_shapes=True)
    model.summary()
    model.compile(
        optimizer='rmsprop',
        loss='mean_squared_error',
        metrics=['mean_squared_error'],
    )
    input_dict = {
        'input_fhv_company':np.array(features['fhv_company']),
        'input_pickup_zone':np.array(features['pickup_zone']),
        'input_request_day_of_week':np.array(features['request_day_of_week']),
        'input_request_hour':np.array(features['request_hour']),
    }
    history = model.fit(
        x=input_dict,
        y=np.array(targets),
        batch_size=128,
        epochs=200,
        verbose=2,
        validation_split=0.25,
        callbacks=tf.keras.callbacks.EarlyStopping(
            patience=10,
            verbose=1,
        ),
    )

    # Extract training and validation loss for plotting
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    plot_train_and_val_loss(train_loss, val_loss)

    # Generate predicted values
    predictions = pd.DataFrame(model.predict(input_dict))
    predictions.columns = ['predicted_driver_pay']
    
    # Create histogram of residuals
    diff = data['driver_pay'] - predictions['predicted_driver_pay']
    diff.hist(bins=50)
    diff.describe()
    
    # Compare predictions and actuals
    compare_df = pd.concat([
        data[categorical_features + ['driver_pay']],
        predictions['predicted_driver_pay'],
    ],
        axis=1,
    )
    fig, ax = plt.subplots(figsize=(8,6))
    sns.histplot(
        x='driver_pay',
        y='predicted_driver_pay',
        data=compare_df,
        bins=50,
        thresh=2,
        cmap='mako',
        cbar=True,
        ax=ax,
    )
    max_val = max(np.reshape(
        compare_df[['driver_pay', 'predicted_driver_pay']].values,
        (-1,)
    ))
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.axline([0,0], [1,1], linestyle='--', color='grey')