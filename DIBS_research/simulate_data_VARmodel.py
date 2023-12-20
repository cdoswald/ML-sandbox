# -*- coding: utf-8 -*-
"""
Date created: 19 Dec 2023
Author: Chris Oswald
Project: DIBS Research
Purpose: Simulate fMRI data using Vector Autoregressive model
"""
# Import packages
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Define constants
n_regions = 50
n_time_periods = 1000
n_burn_in_periods = 100
export_filename = 'DIBS_simulated_fmri_var4model.csv'

# Define functions
def calc_max_abs_eigenvalue(mat_A: np.array) -> float:
    """Calculate maximum absolute eigenvalue for matrix A"""
    return(np.max(np.abs(np.linalg.eigvals(mat_A))))


# Execute main
if __name__ == '__main__':

    # Create t-1 coeff matrix between brain regions
    region_coef_matrix = np.zeros(shape = (n_regions, n_regions))
    for i in range(n_regions):
        for j in range(n_regions):
            if j == i:
                region_coef_matrix[i, j] = 0.9
            elif j > i:
                # Draw from normal distribution (99% of values < 0.99 corr)
                if abs(j - i) <= 1:
                    coef_ij = 0.7
                elif abs(j - i) <= 2:
                    coef_ij = 0.54
                elif abs(j - i) <= 4:
                    coef_ij = 0.43
                else:
                    coef_ij = np.random.normal(loc=0.2, scale=0.05)
                # coef_ij = np.random.normal(loc=0, scale=0.15)
                region_coef_matrix[i, j] = coef_ij
                region_coef_matrix[j, i] = coef_ij

    # Scale coefficient matrices so that time series values don't diverge to inf
    A1 = region_coef_matrix * 0.07 # Decay factor (arbitrary)
    A2 = A1 * 0.02
    A3 = A2 * 0.01
    A4 = A3 * 0.005
    
    A1_max_eig = calc_max_abs_eigenvalue(A1)
    A2_max_eig = calc_max_abs_eigenvalue(A2)
    A3_max_eig = calc_max_abs_eigenvalue(A3)
    A4_max_eig = calc_max_abs_eigenvalue(A4)
    sum_max_eigs = A1_max_eig + A2_max_eig + A3_max_eig + A4_max_eig
    print(f'Sum of max absolute eigenvalues: {sum_max_eigs}') # Diverges if sum >= 1.0
    
    # Initialize vectors with arbitrary scaling
    xt_lag1 = np.ones(shape=(n_regions)) * 20
    xt_lag2 = np.ones(shape=(n_regions)) * 12
    xt_lag3 = np.ones(shape=(n_regions)) * 4
    xt_lag4 = np.ones(shape=(n_regions)) * 1.5
    
    # Create array to save results
    results = np.zeros((n_regions, n_time_periods))
    
    # Specify equation intercept and error term covariance matrix (arbitrary)
    intercept = np.ones(n_regions) * 2
    cov_mat = np.diag(np.ones(n_regions)) * 4
    
    # Generate time series data
    for t in range(n_time_periods + n_burn_in_periods):
        epsilon = np.random.multivariate_normal(np.zeros((n_regions)), cov_mat)
        xt = intercept + A1 @ xt_lag1 + A2 @ xt_lag2 + A3 @ xt_lag3 + A4 @ xt_lag4 + epsilon
        if t >= n_burn_in_periods:
            adj_t = t - n_burn_in_periods
            results[:, adj_t] = xt
        xt_lag4 = xt_lag3
        xt_lag3 = xt_lag2
        xt_lag2 = xt_lag1
        xt_lag1 = xt

    # Plot results
    fig, ax = plt.subplots()
    for i in range(n_regions):
        ax.plot(results[i, :])
    
    # Compute correlation between brain regions
    corr_mat = np.corrcoef(results, rowvar=False)

    # Export data
    os.makedirs('Data', exist_ok=True)
    file_path = os.path.join('Data', export_filename)
    pd.DataFrame(results).to_csv(file_path, index=False)

# =============================================================================
# Archive
# =============================================================================
## Example
# A1 = np.array([
#     [0.2, 0],
#     [-0.3, 0.4]
# ])
# A2 = np.array([
#     [-0.1, 0.1],
#     [0.2, -0.3],
# ])

# Sigma = np.array([
#     [1, 0.7],
#     [0.7, 1],
# ])

# xt_lag1 = np.array([1, 1])
# xt_lag2 = np.array([0.5, 5])


# results = np.zeros((2, 100))
# results[:, 0] = xt_lag2
# results[:, 1] = xt_lag1

# for t in np.arange(2, 100):
#     epsilon = np.random.normal(loc=0, scale=1, size=2) #TODO: correlate error terms
#     epsilon = np.random.multivariate_normal(np.array([0, 0]), Sigma)
#     xt = A1 @ xt_lag1 + A2 @ xt_lag2 + epsilon
#     results[:, t] = xt
#     xt_lag2 = xt_lag1
#     xt_lag1 = xt

# fig, ax = plt.subplots()
# ax.plot(results[0, :])
# ax.plot(results[1, :])
