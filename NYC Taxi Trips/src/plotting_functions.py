# -*- coding: utf-8 -*-
"""
Author:         Chris Oswald
Date Created:   17 August 2023
Project:        NYC Taxi Trips
Purpose:        Create common plotting functions to visualize ML model outputs
"""
# Import packages
from typing import Iterable

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Define functions
def plot_train_and_val_loss(
    train_loss: Iterable,
    val_loss: Iterable,
) -> None:
    """Plot ML model training and validation loss by epoch.
    
    Args
        train_loss: List or other Iterable containing training loss by epoch
        validation_loss: List or other Iterable containing validation loss by epoch
    
    Returns
        None
    """
    fig, ax = plt.subplots(figsize=(6,6))
    x_vals = np.arange(1, len(train_loss) + 1)
    ax.plot(
        x_vals,
        train_loss,
        label='Training Loss',
    )
    ax.plot(
        x_vals,
        val_loss,
        label='Validation Loss',
    )
    ax.legend()
    ax.set_xlabel('Number of Epochs')
    ax.set_ylabel('Loss Metric')