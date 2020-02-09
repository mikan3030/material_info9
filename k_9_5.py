#!/usr/bin/env python
# coding: utf-8

import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Dropout, Activation, Flatten
from keras.layers import Conv2D
from keras import backend as k

np.random.seed(0)
X_train = (np.arange(1,19)).reshape(1,3,3,2)
X_train=X_train.astype(int)
print(X_train)
y_train = np.array([2])

Model = Sequential()

Model.add(Conv2D(1,(2,2),
                # border_mode ="valid",
                use_bias=False,
                input_shape=(3,3,2)))
# Model.add(Dropout(rate=0.25))
# Model.add(Activation("relu"))
# Model.add(Flatten())
# Model.add(Dense(1))
# print(Model.layers)
Get_layer_output = k.function([Model.layers[0].input],
                             [Model.layers[0].output])
# Get_layer_output = k.function([Model.layers[0].input,k.learning_phase()],
#                              [Model.layers[0].output])
# print(Model.get_weights())
# print(Model.get_weights()[0].shape)
# print(Model.get_weights()[1].shape)
Weights = [
    np.zeros((2,2,2,1),dtype="float32"),
    # np.zeros((2,2,2,1),dtype="float32"),
    # np.zeros((1),dtype="float32"),
    # np.zeros((4,1),dtype="float32"),
    # np.zeros((1),dtype="float32")
    ]

Weights[0][0,1,0,0]=-0.1
Weights[0][0,1,1,0]=-0.2
# Weights[1][0] = 5.0
# print(Weights[0])
# print(Weights[0].shape)
Model.set_weights(Weights)
# print(Model.get_weights())
print(Model.summary())

Model.compile(loss="mean_squared_error",
              optimizer ="adadelta",
              metrics=["accuracy"])
out = Get_layer_output([[X_train[0]],0])
# out = Get_layer_output([[X_train[0]],0])
print(out)
print(out[0].shape)
# print(out[0])
print("kusa")
