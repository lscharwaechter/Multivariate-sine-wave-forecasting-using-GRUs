# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 22:39:37 2021

@author: Leon Scharwächter
"""

#import pandas as pd
import torch as th
import numpy as np
import random
from sklearn.preprocessing import MinMaxScaler

# %%

class newDataset(th.utils.data.Dataset):
    """
    A dataset class that can be used with PyTorch's DataLoader
    """
    def __init__(self, n_dim = 8, n_data = 100, sequence_length = 1000):
        """
        Constructor class setting up the data loader
        Generates a dataset in the form
        [n_data, sequence_length, dim]
        return: No return value
        """

        self.n_dim = n_dim
        self.sequence_length = sequence_length

        # Create random sine waves and store them in an array
        N = n_data
        L = sequence_length
        t = np.arange(0, N*L, 1)
        data = np.zeros((len(t), n_dim)) # shape (N * L, n_dim)
        for i in range(n_dim):
            # Each dimension gets a random signal period T
            # -> different frequency
            # and a random Amplitude A
            T = random.randint(10,50)
            A = random.uniform(0.3,0.9)

            data[:,i] = A*np.sin(t/T)

        # Min Max Scaling
        # (For later use, if signals have amplitudes above 1)
        scaler = MinMaxScaler()
        #data = scaler.fit_transform(data)

        # Split data into trainable subsamples
        # (Takes the all sequence_lengths N*L and splits it into
        # n_data N sequence samples)
        dataset = np.array(np.split(data,N))
        # shape (N, L, n_dim)

        self.dataset = dataset
        self.n_sequences = dataset.shape[0] # = n_data (N)
        self.scaler = scaler

    def __len__(self):
        """
        Denotes the total number of samples/sequences that exist
        :return: The number of sequences
        """
        return self.n_sequences # = n_data (N)

    def __getitem__(self, index):
        """
        Generates a sample batch in the form [batch_size, time, dim]
        :return: One batch of data as np.array
        """

        # Load a sample from the dataset and divide it in input and label. The label
        # is the last element of the input to train one step ahead prediction.
        sequence = self.dataset[index] # shape (L, n_dim)
        net_input = np.copy(sequence[:-1,:])
        net_label = np.copy(sequence[-1,:]) # Last time step of all dimensions

        return net_input, net_label
