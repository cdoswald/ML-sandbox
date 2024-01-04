# -*- coding: utf-8 -*-
"""
Date created: 03 January 2024
Author: Chris Oswald
Project: DIBS Research
Purpose: Simulate correlated time series data using Vector Autoregressive (VAR)
         model and Cholesky decomposition
"""
#TODO: test AR(2+) process
#TODO: test non-diagonal F1 matrix

# Import packages
import matplotlib.pyplot as plt
import numpy as np

# Define parameters
np.random.seed(9999)
n_samples = 1000
n_regions = 50

# =============================================================================
# Simulate uncorrelated processes
# =============================================================================
# Define AR(1) process
# Note that if matrix is not diagonal, then the value for each region depends
# both on the value for that region's previous value as well as the values for
# the other regions' previous values
F1 = np.diag(np.ones(n_regions)) * np.random.uniform(0, 1, size=n_regions)

# Define error term mean vector and covariance matrix (i.e., variance of each error term)
# Note that off-diagonal values are zero since we assume that covariance between
# error terms for two different brain regions is 0 (i.e., the errors are independent)
error_mean_vec = np.ones(n_regions) * 0
error_cov_mat = np.diag(np.ones(n_regions)) * np.abs(np.random.normal(2, 1, size=n_regions))
print(error_cov_mat)

## Create an (n_region * n_sample) matrix of error values
Q = np.random.multivariate_normal(error_mean_vec, error_cov_mat, size=n_samples).T
print(f'Error matrix shape: {Q.shape}')

# Create blank (n_region * n_sample) matrix of values
Z = np.zeros((n_regions, n_samples))
print(f'Process matrix shape: {Z.shape}')

# Simulate values for time period i
for i in range(1, n_samples):
    Z[:, i] = F1 @ Z[:, i-1] + Q[:, i]

# Plot uncorrelated processes and calculate max empirical correlation
fig, ax = plt.subplots(figsize=(10, 6))
for i in range(n_regions):
    ax.plot(Z[i], label=f'Uncorr. Process {i}')
empirical_corr_mat_Z = np.corrcoef(Z)
max_empir_corr_Z = max(empirical_corr_mat_Z[abs((empirical_corr_mat_Z - 1)) > 0.005])

# =============================================================================
# Use Cholesky decomposition to induce correlation between processes
# =============================================================================
# Define correlation matrix between processes
region_corr_mat = np.eye(n_regions)
for i in range(n_regions):
    for j in range(i+1, n_regions):
        # corr_coef_mean = ((n_regions - abs(i - j)) / n_regions)
        # coef_ij = np.random.normal(loc=corr_coef_mean, scale=0.05)
        coef_ij = np.random.uniform(0.75, 0.9)  # * (n_regions - abs(i - j)) / n_regions
        region_corr_mat[i, j] = coef_ij
        region_corr_mat[j, i] = coef_ij

# Modify specified correlation matrix to ensure positive semi-definiteness
eigenvals, eigenvecs = np.linalg.eig(region_corr_mat)
Lambda = np.diag(np.maximum(eigenvals, 0.005))
psd_corr_mat = eigenvecs @ Lambda @ np.linalg.inv(eigenvecs)

# Convert correlation matrix to covariance matrix
# Note that "D" is a diagonal matrix of standard deviations
D = np.sqrt(error_cov_mat)
Sigma = D @ psd_corr_mat @ D

# Generate Cholesky decomposition of covariance matrix
A = np.linalg.cholesky(Sigma)

# Define intercept vector
u = np.random.normal(20, 1, size=(n_regions, 1))

# Multiply Cholesky decomposition matrix A by uncorrelated process matrix Z to 
# induce correlation between processes
X = A @ Z + u

# Plot correlated processes
fig, ax = plt.subplots(figsize=(10, 6))
for i in range(n_regions):
    ax.plot(X[i], label=f'Corr. Process {i}')
empirical_corr_mat_X = np.corrcoef(X)
max_empir_corr_X = max(empirical_corr_mat_X[abs((empirical_corr_mat_X - 1)) > 0.005])

# Compute difference between empirical correlation matrix and specified corr matrix
corr_diff_psd = abs(psd_corr_mat - region_corr_mat)
print(max(corr_diff_psd.reshape(-1)))
corr_diff_emp = abs(empirical_corr_mat_X - region_corr_mat)
print(max(corr_diff_emp.reshape(-1)))
