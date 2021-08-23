import json
import keras
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten, Dropout
from keras.models import Sequential
import numpy as np
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier


def create_model():
    model = Sequential()
    model.add(Convolution2D(
        # batch_input_shape=(64, 1, 28, 28),
        filters=20,
        kernel_size=5,
        strides=1,
        input_shape=[25, 50, 3],
    ))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(
        pool_size=[2, 2],
        strides=[2, 2],
    ))

    model.add(Flatten())
    model.add(Dropout(0.2))

    # 2 output x,y
    model.add(Dense(units=2, activation='tanh'))

    model.compile(optimizer=keras.optimizers.Adam(0.0005), loss='mean_squared_error', metrics=['accuracy'])

    return model
