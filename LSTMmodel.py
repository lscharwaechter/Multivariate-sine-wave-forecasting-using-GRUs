# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 16:39:30 2021

@author: Leon Scharwächter
"""

import torch.nn as nn

# LSTM input shape:  (N, L, H_{in}) for batch_first=True
# N = batch size
# L = sequence length
# Hin = input size (feature dimensions)

class LSTMmodel(nn.Module):

    def __init__(self, n_features=8, n_hidden=128, n_layers=2):
        super().__init__()

        self.n_hidden = n_hidden
        self.n_features = n_features

        self.lstm = nn.LSTM(
                input_size = n_features,
                hidden_size = n_hidden,
                num_layers = n_layers,
                batch_first = True,
                dropout = 0.2
                )

        self.linear = nn.Linear(
                n_hidden,
                n_features)

    def forward(self, x, state=None):
        self.lstm.flatten_parameters()

        if state == None:
            x, (h, c) = self.lstm(x)
        else:
            x, (h, c) = self.lstm(x, state)
        # x now has the shape [N, L, D∗Hout]

        # For passing only the last time step to the linear layer:
        x = x[:,-1,:]
        x = self.linear(x)
        return x, (h, c)