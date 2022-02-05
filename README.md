# Multivariate Sine-wave Forecasting using GRUs
This is a running fun project to predict the waveform of multiple, generated sine-waves using Gated Recurrent Units (GRUs) in PyTorch. loadData.py creates the dataset, consisting of K x N x L dimensions where K is the number of sine wave types, which randomly differ in their frequency and amplitude, N is the number of samples for each sine wave and L is the sequence length per sample. GRUmodel.py defines the underlying neural network based on GRU- and linear layers. In GRUtrain.py the model is fitted gradient-based to the dataset using MSE as the Loss Function and currently Adam with Weight Decay (AdamW) as Optimization.

In the plot, Teacher-Forcing is done for 1000 time steps and afterwards the model runs in a Closed-Loop:

![Plots_GRU](https://user-images.githubusercontent.com/56418155/152622737-072afa14-fd8b-4e64-92c5-47d3c12665d2.png)

