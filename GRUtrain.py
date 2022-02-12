# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 16:42:23 2021

@author: Leon Scharwächter
"""

from GRUmodel import GRUmodel
from loadData import newDataset

import torch as th
import numpy as np
import matplotlib.pyplot as plt
import os

# %% Settings

# Model
n_hidden = 128

# Dataset
n_sinetypes = 1 #3
n_data = 150
sequence_length = 250

# Training
validation_split = .25
batch_size = 16
learning_rate = 0.001
num_epochs = 60 

# Save To File
model_save = 0
model_name = "william"

# Forecasting
n_examples = 3 # How many batches to visualize
tf_steps = 250 # Teacher Forcing steps
cl_steps = 3000 # Closed Loop Steps


# %% Load the model and dataset

GRUmodel = GRUmodel(
        n_features = n_sinetypes,
        n_hidden = n_hidden,
        n_layers = n_sinetypes 
        )

dataset = newDataset(
        n_dim = n_sinetypes,
        n_data = n_data,
        sequence_length = sequence_length
        )

# %% Split dataset into training and validation set

def buildTrainingTestSet(dataset, validation_split, batch_size):
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
    
    return train_loader, validation_loader

# %% Build the training procedure 

def train(GRUmodel, train_loader, validation_loader, batch_size,
          learning_rate, num_epochs):
    
    print("Training...")
    optimizer = th.optim.AdamW(GRUmodel.parameters(),
                               lr=learning_rate)    
    criterion = th.nn.MSELoss()
    
    # Set up a list to save and store the epoch errors
    epoch_errors = []
    epoch_errors_val = []
    best_error = np.infty
    best_epoch = 0

    for epoch in range(num_epochs):
        # List to store the errors for each sequence
        sequence_errors = []
        sequence_errors_val = []
        #Train:
        for batch_index, (net_input, net_label) in enumerate(train_loader):
            # net_input shape: [batch_size, sequence_length-1, dim]
            # net_label shape: [batch_size, dim]
    
            # Reset optimizer to clear the previous batch
            optimizer.zero_grad()
    
            # PyTorch needs Tensors of type float as input
            net_input = net_input.float()
            net_label = net_label.float()
    
            # Prediction using a batch of sequence samples
            yhat,_ = GRUmodel(net_input) # yhat shape [batch_size, dim]
    
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
        print("Epoch: ", epoch+1, "/", num_epochs)
        print("Average Training Error: ", epoch_errors[-1])
    
        # Get Test Error
        for batch_index2, (net_input2, net_label2) in enumerate(validation_loader):
            optimizer.zero_grad()
            net_input2 = net_input2.float()
            net_label2 = net_label2.float()
            yhat2,_ = GRUmodel(net_input2)
            loss2 = criterion(yhat2, net_label2)
            sequence_errors_val.append(loss2.item())
        epoch_errors_val.append(np.mean(sequence_errors_val))
        print("Average Test Error: ", epoch_errors_val[-1])
        
        if best_error > epoch_errors_val[-1]:
            best_error = epoch_errors_val[-1]
            best_weights = GRUmodel.state_dict()
            best_epoch = epoch
    
    # Plot Train/Test Error
    plt.figure()
    plt.plot(epoch_errors)
    plt.plot(epoch_errors_val)
    plt.legend(["Train Error","Test Error"])
    plt.xlabel("Epochs")
    plt.ylabel("Error")
    print('\nBest Epoch:',best_epoch)
    print('Average Test Error:',best_error)
    
    # Load the weights from the best epoch
    GRUmodel.load_state_dict(best_weights)
    
    # Save the weights of the model into a file
    if model_save:
        th.save(best_weights,
            os.path.join(os.getcwd(),model_name + ".pt"))


# %%  Feed the sine waves into the model and let it continue the signals

def forecasting(GRUmodel, validation_loader, n_examples, tf_steps, cl_steps):
    
    print("\nPlotting...")
    
    # Load the saved model state
    #if model_save:
    #    GRUmodel.load_state_dict(
    #            th.load(os.path.join(os.getcwd(),model_name + ".pt")))
    
    for batch in range(n_examples):
        net_input, net_label = next(iter(validation_loader))
        net_input = net_input.float()
        net_label = net_label.float()
    
        # Extract the first Steps for Teacher Forcing
        x = net_input[:,:tf_steps,:] # [1, tf_steps, dim]
    
        # Start with the Teacher Forcing Steps
        yhat, h = GRUmodel(x)
        yhat = yhat[np.newaxis,:,:] # [batch_size = 1, 1, dim]
        # Append the model output to the input and...
        x = th.cat((x, yhat.float()), dim=1) # [batch_size = 1, tf_steps + 1, dim]
    
        # ...continue with the Closed Loop Steps
        for t in range(cl_steps):
            # Generate a prediction with the current hidden- and cell state
            yhat, h = GRUmodel(x=yhat, state=h)
            # Append the model output to the input and continue the loop
            yhat = yhat[np.newaxis,:,:] # new codeline
            x = th.cat((x, yhat.float()), dim=1)
    
        x_plot = x.detach().numpy()
        
        # Inverse Transform is only needed if in the preprocessing of loadData.py 
        # the data is normalized to Min-Max-Scaling
        #x_plot = dataset.scaler.inverse_transform(x_plot[0,:,:])[np.newaxis,:,:]
    
        plt.figure()
        plt.plot(x_plot[0,:,:])
        plt.vlines(tf_steps-1, ymin = np.min(x_plot), ymax = np.max(x_plot), colors='r')
    
    print("Done")

# %% Procedure

train_loader, validation_loader = buildTrainingTestSet(dataset, validation_split, batch_size)

train(GRUmodel, train_loader, validation_loader, batch_size, learning_rate, num_epochs)
forecasting(GRUmodel, validation_loader, n_examples, tf_steps, cl_steps)
