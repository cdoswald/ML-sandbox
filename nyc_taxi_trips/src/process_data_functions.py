# -*- coding: utf-8 -*-
"""
Author: Chris Oswald
Date Created: 16 July 2023
Project: NYC Taxi Trips
"""
# Import packages
from typing import Dict

import numpy as np
import pandas as pd

# Define functions
def truncate_numerical_vars(
    df: pd.DataFrame,
    truncation_dict: Dict[str, Dict[str, float]],
    drop_obs: bool = True,
) -> pd.DataFrame:
    """Truncate numerical variables in dataframe.

    Args:
        df: pd.DataFrame containing one or more variables
        truncation_dict: nested dictionary of form
            {
                "var_name_1":{
                    "min":value,
                    "max":value,
                },
                "var_name_2":{...
                },
            }
        drop_obs: boolean indicating whether records should be removed
            from dataframe (default = True); if False, values will still be
            truncated but observations will be retained

    Returns
        pd.DataFrame with truncated variables
    """
    for var, var_dict in truncation_dict.items():
        if var not in df.columns:
            continue
        min_val = float(var_dict['min'])
        max_val = float(var_dict['max'])
        if drop_obs:
            df = df.loc[((df[var] >= min_val) & (df[var] <= max_val))]
        else:
            df[var] = np.where(df[var] < min_val, min_val, df[var])
            df[var] = np.where(df[var] > max_val, max_val, df[var])
    return df.reset_index(drop=True)
