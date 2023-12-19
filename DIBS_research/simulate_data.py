# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 17:54:57 2023

@author: chris
"""
# Import packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import truncexpon

# Define constants
n_regions = 50
n_time_periods = 100

# Define functions
def get_response(probs: np.array, corr_mat: np.array) -> np.array:
    """Draw activation value for each region for time period t.

    Args
        probs: numpy array of probabilities to determine main activated region
        corr_mat: numpy array indicating correlation for (n_regions * n_regions)

    Returns
        numpy array of activation value for each region
    """
    ## Select one region for main activation (using non-uniform baseline probability)
    activ_region = np.random.choice(np.arange(0, n_regions), p=probs)
    ## Select activation amount (uniform) for selected region
    activ_value = np.random.uniform(low=4, high=10)
    ## Calculate response of other regions using correlation matrix
    response = activ_value * corr_mat[activ_region, :]
    response = np.where(response < 0, 0, response) # Truncate response at 0
    return(response)


# Execute main
if __name__ == '__main__':

    # Create correlation matrix between brain regions
    region_corr_matrix = np.zeros(shape = (n_regions, n_regions))
    for i in range(n_regions):
        for j in range(n_regions):
            if j == i:
                region_corr_matrix[i, j] = 1
            elif j > i:
                # Draw from normal distribution (99% of values < 0.99 corr)
                corr_ij = np.random.normal(loc=0.6, scale=0.13)                 #TODO: add structure
                region_corr_matrix[i, j] = corr_ij
                region_corr_matrix[j, i] = corr_ij

    # Determine baseline probability that any particular region is activated 
    # at time t
    probs_unnorm = np.random.uniform(0, 1, size=50)                             #TODO: add structure
    probs_norm = probs_unnorm / sum(probs_unnorm)
    probs_norm[-1] = 1 - sum(probs_norm[:-1])

    # Model exponential decay of activation from time period t to t+1
    ## Each region has decay rate drawn from truncated exponential distribution
    ## so that the value is in the range [0,1] (i.e., activation always goes to 0
    ## over time)
    base_lambda = 1                                                             #TODO: add structure/group closer
    decay_coeffs = truncexpon.rvs(base_lambda, size=n_regions)
    sns.histplot(decay_coeffs)

    # Simulate all time periods
    results = np.zeros(shape=(n_regions, n_time_periods))
    results[:, 0] = get_response(probs_norm, region_corr_matrix)
    sns.scatterplot(results[:, 0])

    for t in np.arange(1, n_time_periods):
        # Compute residual response from previous activations
        results[:, t] = results[:, (t-1)] * decay_coeffs 
        if (t % 50 == 0):                                                        #TODO: reduce structure
            # Compute new activation
            results[:, t] += get_response(probs_norm, region_corr_matrix)       #TODO: randomly select subset/add noise
    results = pd.DataFrame(results)

    # Plot results
    fig, ax = plt.subplots()
    for i in results.index:
        ax.plot(results.loc[i, ])
