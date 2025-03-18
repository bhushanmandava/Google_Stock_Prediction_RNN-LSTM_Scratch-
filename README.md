# Stock Price Prediction using LSTM

This project is a deep learning-based stock price prediction model using Long Short-Term Memory (LSTM) networks implemented in PyTorch. The model is trained on historical stock price data to forecast future stock prices.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction
Stock price prediction is a challenging task due to the stochastic nature of financial markets. This project implements an LSTM-based neural network to analyze and predict stock prices using past trends.

## Features
- Data preprocessing (scaling and transformation)
- LSTM-based neural network
- Training with PyTorch
- Model evaluation and visualization
- Predictions on test data



## Dataset
The dataset consists of historical stock prices, including features such as opening price, closing price, high, low, and volume.



## Model Architecture
- Input Layer: Process time-series stock data
- LSTM Layers: Extract temporal dependencies
- Fully Connected Layer: Predict stock prices

## Training
The model is trained using Mean Squared Error (MSE) loss and Adam optimizer with a specified number of epochs.

## Evaluation
Model performance is evaluated using:
- Mean Squared Error (MSE)
- Visualization of predicted vs actual stock prices

## Results
The trained model produces accurate short-term stock price predictions. The performance improves with more training data and hyperparameter tuning.

## Contributing
Contributions are welcome! Feel free to open an issue or submit a pull request.

