# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 21:32:25 2023

@author: chris
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
        train_loss:
        validation_loss:
    
    Returns
        None
    """
    fig, ax = plt.subplots(figsize=(12,9))
    x_vals = np.arange(1, len(train_loss) + 1)
    sns.scatterplot(
        x=x_vals,
        y=train_loss,
        ax=ax,
    )
    sns.scatterplot(
        x=x_vals,
        y=val_loss,
        ax=ax,
    )
