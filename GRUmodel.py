# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 16:39:30 2021

@author: Leon Scharwächter
"""

import torch.nn as nn

# GRU input shape:  (N, L, H_in) for batch_first=True
# N = batch size
# L = sequence length
# H_in = input size (feature dimensions

class GRUmodel(nn.Module):

    def __init__(self, n_features=8, n_hidden=128, n_layers=2):
        super().__init__()

        self.n_hidden = n_hidden
        self.n_features = n_features

        self.gru = nn.GRU(
                input_size = n_features,
                hidden_size = n_hidden,
                num_layers = n_layers,
                batch_first = True,
                #dropout = 0.2,
                #bidirectional = True
                )

        self.linear = nn.Linear(
                n_hidden,
                n_features,
                bias=True)

    def forward(self, x, state=None):
        self.gru.flatten_parameters()

        if state == None:
            x, h = self.gru(x) # x, (h, c)
        else:
            x, h = self.gru(x, state)
        # x now has the shape [N, L, D∗Hout resp. n_hidden]
        # or [N, L, 2*n_hidden] if bidirectional

        # if bidirectional = True, sum up both directions
        #x_sum = x[:,:,:self.n_hidden] + x[:,:,self.n_hidden:]
        x_sum = x

        # For passing only the last time step to the linear layer
        x_sum = x_sum[:,-1,:]
        x_sum = self.linear(x_sum)
        return x_sum, h
