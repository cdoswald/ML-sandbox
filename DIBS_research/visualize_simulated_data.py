# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 20:04:08 2024

@author: chris
"""
# Import packages
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Specify directory/file paths
data_dir = "Data"
image_dir = "Images"
os.makedirs(image_dir, exist_ok=True)

# sim_data_file = "DIBS_simulated_fmri_var4model.csv"
sim_data_file = "DIBS_simulated_fmri_VAR1_corr_model.csv"

# Load data
data = pd.read_csv(os.path.join(data_dir, sim_data_file), header=0)
data.head()

# Plot all results
fig, ax = plt.subplots(figsize=(16,12))
for i in np.arange(1, len(data)+1, 5):
    ax.plot(data.iloc[i, 1:])
ax.set_xticks(np.arange(0, 1001, 100))
fig.savefig(os.path.join(image_dir, "DIBS_simulated_data_all_regions.png"))

data = data.reset_index(drop=False).rename(columns={"index":"region_id"})
data.head()

long_data = pd.melt(data, id_vars="region_id", var_name="time_index")
long_data[['region_id', 'time_index']] = long_data[['region_id', 'time_index']].astype('int')
long_data

data_subset = long_data.loc[
    (long_data['region_id'] < 25) & 
    (long_data['time_index'] < 200)
]
g = sns.FacetGrid(data_subset, col="region_id", col_wrap=5)
g.map_dataframe(sns.lineplot, x="time_index", y="value")
g.savefig(os.path.join(image_dir, 'DIBS_simulated_data_by_region.png'))