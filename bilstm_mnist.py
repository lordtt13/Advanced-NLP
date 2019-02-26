# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 01:18:45 2019

@author: tanma
"""

import os
from keras.models import Model
from keras.layers import Input, CuDNNLSTM, CuDNNGRU, Bidirectional
from keras.layers import GlobalMaxPooling1D, Lambda, Concatenate, Dense
from keras.datasets import mnist
import keras.backend as K
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


(x_train, y_train),(x_test,y_test) = mnist.load_data()

D = 28
M = 15

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

input_ = Input(shape=(D, D))

rnn1 = Bidirectional(CuDNNLSTM(M, return_sequences=True))
x1 = rnn1(input_) 
x1 = GlobalMaxPooling1D()(x1) 

rnn2 = Bidirectional(CuDNNLSTM(M, return_sequences=True))  

permutor = Lambda(lambda t: K.permute_dimensions(t, pattern=(0, 2, 1)))

x2 = permutor(input_)
x2 = rnn2(x2) 
x2 = GlobalMaxPooling1D()(x2) 

concatenator = Concatenate(axis=1)
x = concatenator([x1, x2]) 

output = Dense(10, activation='softmax')(x)

model = Model(inputs=input_, outputs=output)

model.compile(
  loss='sparse_categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)

model.fit(x_train, y_train, batch_size=32, epochs=10, validation_split = 0.25)

