# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 16:42:23 2021

@author: Leon Scharwächter
"""

from LSTMmodel import LSTMmodel
from loadData import newDataset

import torch as th
import numpy as np
import matplotlib.pyplot as plt
import os

# %% Load the model and dataset

LSTMmodel = LSTMmodel(
        n_features = 8,
        n_hidden = 64,
        n_layers = 2
        )

dataset = newDataset(
        n_dim = 8,
        sequence_length = 1000
        )

# %% Model settings

model_save = 1
model_name = "william"

# %% Split dataset into training and validation set

validation_split = .25
batch_size = 16
shuffle_dataset = True

# Creating data indices for training and validation splits:
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))

if shuffle_dataset:
    #np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating data samplers...
train_sampler = th.utils.data.SubsetRandomSampler(train_indices)
valid_sampler = th.utils.data.SubsetRandomSampler(val_indices)

# ...and loaders
train_loader = th.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        drop_last=True)

validation_loader = th.utils.data.DataLoader(
        dataset=dataset,
        batch_size=1,
        sampler=valid_sampler,
        drop_last=True)

# Number of batches:
#print(len(train_loader))
#print(len(validation_loader))

# %% Train the model
print("Training...")

optimizer = th.optim.AdamW(LSTMmodel.parameters(),
                           lr=0.001)
criterion = th.nn.MSELoss()

# Set up a list to save and store the epoch errors
epoch_errors = []
best_error = np.infty

num_epochs = 500
for epoch in range(num_epochs):
    # List to store the errors for each sequence
    sequence_errors = []
    #Train:
    for batch_index, (net_input, net_label) in enumerate(train_loader):
        # net_input and net_label have shape: [batch_size, sequence_length-1, dim]
        # Reset optimizer to clear the previous batch
        optimizer.zero_grad()
        # PyTorch needs Tensors of type float as input
        net_input = net_input.float()
        net_label = net_label.float()
        # Prediction using one sample sequence
        yhat,_ = LSTMmodel(net_input)
        # Compute Loss
        loss = criterion(yhat, net_label)
        # Compute gradients
        loss.backward()
        # Perform weight update
        optimizer.step()
        # Save error
        sequence_errors.append(loss.item())
    # Calculate the mean error of the epoch
    epoch_errors.append(np.mean(sequence_errors))
    print("Epoch: ", epoch+1)
    print("Average Training Error: ", epoch_errors[-1])

# Save the trainable parameters of the model
if model_save:
    th.save(LSTMmodel.state_dict(),
        os.path.join(os.getcwd(),model_name + ".pt"))

# %% Testing
print("Testing...")

LSTMmodel.load_state_dict(
        th.load(os.path.join(os.getcwd(),model_name + ".pt")))


tf_steps = 1000 # Teacher Forcing steps
cl_steps = 1000 # Closed Loop Steps

# How many batches to visualize
examples = 5

for batch in range(examples):
    net_input, net_label = next(iter(validation_loader))
    net_input = net_input.float()
    net_label = net_label.float()

    # Extract the first Steps for Teacher Forcing
    x = net_input[:,:tf_steps,:] # [1, tf_steps, dim]

    # Start with the Teacher Forcing Steps
    yhat, (h, c) = LSTMmodel(x)
    yhat = yhat[np.newaxis,:,:] # [batch_size = 1, 1, dim]
    # Append the model output to the input and...
    x = th.cat((x, yhat.float()), dim=1) # [batch_size = 1, tf_steps + 1, dim]

    # ...continue with the Closed Loop Steps
    for t in range(cl_steps):
        # Generate a prediction with the current hidden- and cell state
        yhat, (h, c) = LSTMmodel(x=yhat, state=(h, c))
        # Append the model output to the input and continue the loop
        yhat = yhat[np.newaxis,:,:] # new codeline
        x = th.cat((x, yhat.float()), dim=1)

    x_plot = x.detach().numpy()
    x_plot = dataset.scaler.inverse_transform(x_plot[0,:,:])[np.newaxis,:,:]

    plt.figure()
    plt.plot(x_plot[0,:,:])
    plt.vlines(tf_steps-1, ymin = np.min(x_plot), ymax = np.max(x_plot), colors='r')

print("Done")

# %%
