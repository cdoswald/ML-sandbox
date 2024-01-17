# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 20:02:31 2024

@author: chris
"""
# Import packages
import json
import os
from typing import Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

# Define classes
class CustomDataset(Dataset):
    """PyTorch custom dataset class."""

    def __init__(self, features, labels):
        self.features = torch.from_numpy(np.array(features)).float()
        self.labels = torch.from_numpy(np.array(labels)).float()
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return (self.features[idx], self.labels[idx])

    # Convert sequences to PyTorch tensors
    # train_input_tensor = torch.from_numpy(np.array(train_input_seq)).float().unsqueeze(-1)
    # train_input_tensor = train_input_tensor.to(device) # Note that "to" method for tensors is not in-place operation (like it is for nn.Module/model below)
    # train_input_tensor.shape
    
    # train_target_tensor = torch.from_numpy(np.array(train_target_seq)).float().unsqueeze(-1)
    # train_target_tensor = train_target_tensor.to(device) # Note that "to" method for tensors is not in-place operation (like it is for nn.Module/model below)
    # train_target_tensor.shape
class RNN(nn.Module):
    "Recurrent neural network class."

    def __init__(self, input_size, hidden_size, output_size, n_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers=n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        
        batch_size = x.size(0)
        hidden = (
            torch.zeros(self.n_layers, batch_size, self.hidden_size).to(device),
            torch.zeros(self.n_layers, batch_size, self.hidden_size).to(device)
        )

        out, hidden = self.rnn(x, hidden)

        out = self.fc(out[:, -1, :])

        return out, hidden


# Define functions
def split_train_validate_test(
    data: pd.DataFrame,
    probs: Optional[Iterable] = [0.5, 0.25, 0.25],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataframe into training, validation, and test datasets according
    to the probabilities provided.
    
    Arguments
        data: dataframe containing original data
        probs: vector of training, validation, and test shares

    Returns
        tuple containing (train_data, validate_data, test_data)
    """
    assign_vec = np.random.choice(['train', 'val', 'test'], len(data), p=probs)
    train_data = data.loc[assign_vec == 'train'].reset_index(drop=True)
    validate_data = data.loc[assign_vec == 'val'].reset_index(drop=True)
    test_data = data.loc[assign_vec == 'test'].reset_index(drop=True)
    return(train_data, validate_data, test_data)


def create_input_and_target_sequences(
    data: pd.DataFrame,
    seq_length: Optional[int] = 100,
) -> Tuple[List, List]:
    """For each time series, using rolling window to create additional
    observations of boths inputs and targets of specified sequence length.
    
    Arguments
        data: dataframe containing training, validation, or test data
        seq_length: sequence length integer indicating length of rolling window
    
    Returns
        tuple containing (input_sequences, input_targets)
    """
    input_seqs = []
    targets = []
    for obs in range(len(data)):
        obs_data = data.iloc[obs]
        for i in range(len(obs_data) - seq_len):
            input_seqs.append(obs_data[i:i+seq_len].values)
            targets.append(obs_data[i+seq_len])
            i += 1
    return(input_seqs, targets)

        
if __name__ == '__main__':
    
    # Define constants
    batch_size = 32
    seq_len = 100
    
    min_response_val = 20 # (Used for normalization)
    max_response_val = 50 # (Used for normalization)
    
    torch.manual_seed(9999)
    np.random.seed(9998)
    
    # Specify directory/file paths
    data_dir = "Data"
    # sim_data_file = "DIBS_simulated_fmri_var4model.csv"
    sim_data_file = "DIBS_simulated_fmri_VAR1_corr_model.csv"
    
    # Check that CUDA GPUs are available
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        print(f'Number of GPUs available: {torch.cuda.device_count()}')
        for i in range(torch.cuda.device_count()):
            print(f'GPU {i+1}: {torch.cuda.get_device_name(i)}')
    else:
        device = torch.device("cpu")
        print("GPU not available--using CPU instead.")
    
    # Load data
    data = pd.read_csv(os.path.join(data_dir, sim_data_file), header=0)
    data.head()
    
    # Normalize data
    data = (data - min_response_val)/(max_response_val - min_response_val)
    
    # Shuffle data
    shuffled_idx = np.random.permutation(len(data))
    data = data.loc[shuffled_idx].reset_index(drop=True)
    
    # Split data
    train, validate, test = split_train_validate_test(data)
    print(f'Train: {train.shape} \nValid: {validate.shape} \nTest: {test.shape}')

    # Create input and target sequences
    train_inputs, train_targets = create_input_and_target_sequences(train, seq_len)
    valid_inputs, valid_targets = create_input_and_target_sequences(validate, seq_len)
    test_inputs, test_targets = create_input_and_target_sequences(test, seq_len)
    print(f"Train: # inputs: {len(train_inputs)}, # targets: {len(train_targets)}")
    print(f"Valid: # inputs: {len(valid_inputs)}, # targets: {len(valid_targets)}")
    print(f"Test: # inputs: {len(test_inputs)}, # targets: {len(test_targets)}")
    
    # Convert inputs and targets to PyTorch datasets
    train_dataset = CustomDataset(train_inputs, train_targets)
    valid_dataset = CustomDataset(valid_inputs, valid_targets)
    test_dataset = CustomDataset(test_inputs, test_targets)
    
    # Create dataloaders
    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dl = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # Instantiate model
    input_size = 1
    hidden_size = 64
    output_size = 1
    
    model = RNN(input_size, hidden_size, output_size, n_layers=1)
    model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train model
    num_epochs = 100
    epoch_results = {}
    for epoch in range(num_epochs):

        # Training data
        train_step_loss = []
        model.train()
        for batch_idx, (inputs, targets) in enumerate(train_dl):
            outputs, hidden = model(inputs.unsqueeze(-1).to(device))
            train_loss = criterion(outputs, targets.unsqueeze(-1).to(device))
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            train_step_loss.append(train_loss.item())
        
        # Validation data
        model.eval()
        valid_step_loss = []
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(valid_dl):
                outputs, _ = model(inputs.unsqueeze(-1).to(device))
                valid_loss = criterion(outputs, targets.unsqueeze(-1).to(device))
                valid_step_loss.append(valid_loss.item())
        
        # Record average loss on train and validation data for each epoch
        epoch_results[epoch] = {
            'train_loss':np.mean(train_step_loss),
            'valid_loss':np.mean(valid_step_loss),
        }
        print(f'Epoch: {epoch+1}/{num_epochs}, Loss: {np.mean(train_step_loss):.4f}')

    with open('epoch_results.json', 'w') as io:
        json.dump(epoch_results, io, indent=4)

#     # Generate predicted values for training data
#     model.eval()
#     with torch.no_grad():
#         train_predictions, _ = model(train_input_tensor)
#     train_predictions
    
#     # Compute MSE for training data
#     train_target_tensor = train_target_tensor.cpu()
#     train_predictions = train_predictions.cpu()
    
#     train_mse = (sum((train_target_tensor - train_predictions)**2))/len(train_predictions)
#     train_mse
    
#     # Plot train predictions against actuals
#     fig, ax = plt.subplots()
#     ax.scatter(
#         x=train_target_tensor.cpu(),
#         y=train_predictions.cpu(),
#         alpha=0.1
#     )
#     ax.axline([0, 0], [1, 1], linestyle="--")
    
#     # Create input and target sequences for test data
#     test_input_seq = []
#     test_target_seq = []
    
#     for obs in range(len(test_data)):
#         obs_data = test_data.iloc[obs]
#         for i in range(len(obs_data) - seq_len):
#             test_input_seq.append(obs_data[i:i+seq_len].values)
#             test_target_seq.append(obs_data[i+seq_len])
#             i += 1
#     print("# test input seqs: ", len(test_input_seq), "\n# test target seqs: ", len(test_target_seq))
    
#     # Convert test sequences to PyTorch tensors
#     test_input_tensor = torch.from_numpy(np.array(test_input_seq)).float().unsqueeze(-1)
#     test_input_tensor = test_input_tensor.to(device) # Note that "to" method for tensors is not in-place operation (like it is for nn.Module/model below)
#     test_input_tensor.shape
    
#     test_target_tensor = torch.from_numpy(np.array(test_target_seq)).float().unsqueeze(-1)
#     test_target_tensor = test_target_tensor.to(device) # Note that "to" method for tensors is not in-place operation (like it is for nn.Module/model below)
#     test_target_tensor.shape
    
#     # Generate predictions for test data
#     model.eval()
#     with torch.no_grad():
#         test_predictions, _ = model(test_input_tensor)
#     test_predictions
    
#     # Compute MSE for test data
#     test_target_tensor = test_target_tensor.cpu()
#     test_predictions = test_predictions.cpu()
    
#     test_mse = (sum((test_target_tensor - test_predictions)**2))/len(test_predictions)
#     test_mse
    
#     # Plot test predictions against actuals
#     fig, ax = plt.subplots()
#     ax.scatter(
#         x=test_target_tensor.cpu(),
#         y=test_predictions.cpu(),
#         alpha=0.1
# )
# ax.axline([0, 0], [1, 1], linestyle="--")
# ax.set_xlim(-1, 1)
# ax.set_ylim(-1, 1)
