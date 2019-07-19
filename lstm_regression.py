# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 12:35:21 2019

@author: tanma
"""

import pandas as pd,numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import CuDNNLSTM as LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import test_train_split

data = pd.read_csv("airline_passengerrs.csv").values

np.random.seed(0)

scaler = MinMaxScaler()
data = scaler.fit_transform(data)

data_train,data_test = test_train_split(test_size = 0.75)

