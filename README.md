# Multivariate Sine-wave Forecasting using LSTMs
This is a running fun project to predict the waveform of multiple, generated sine-waves using LSTMs in PyTorch. loadData.py creates the dataset, consisting of N x K x L dimensions where N is the number of sine wave types, which randomly differ in their frequency and amplitude, K is the number of samples of each sine wave and L is the sequence length per sample. LSTMmodel.py defines the underlying neural network based on LSTM- and linear layers.

![Plots](https://user-images.githubusercontent.com/56418155/152160733-00fbf8ad-90df-4e20-ad87-a4204edf6cc9.png)
