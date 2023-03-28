import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv1D, BatchNormalization, MaxPooling1D


def autoEncoderNeuralNetwork():
    # design an autoEncoder neural network classifier for anomaly detection
    # the autoEncoder is distributed into encoder and decoder
    model = Sequential()
    # Encoder Structure
    model.add(Conv1D(filters=64, kernel_size=6, activation='relu',
                     padding='same', input_shape=(72, 1)))
    model.add(BatchNormalization())
    # adding a pooling layer
    model.add(MaxPooling1D(pool_size=3, strides=2, padding='same'))

    model.add(Conv1D(filters=64, kernel_size=6, activation='relu',
                     padding='same', input_shape=(72, 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=3, strides=2, padding='same'))

    model.add(Conv1D(filters=64, kernel_size=6, activation='relu',
                     padding='same', input_shape=(72, 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=3, strides=2, padding='same'))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    return model

