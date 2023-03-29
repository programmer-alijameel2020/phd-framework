import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv1D, BatchNormalization, MaxPooling1D


def modelConstruction(LayerArray):
    # design an autoEncoder neural network classifier for anomaly detection
    # LayerArray: is the parameter that contains the layer types and configurations
    model = Sequential()
    for layer in LayerArray:
        model.add(layer)

    return model


def initializeLayerArray():
    layerArray = []
    # adding the processing layers to the layer array to be forwarded for construction
    layerArray.append(Conv1D(filters=64, kernel_size=6, activation='relu', padding='same', input_shape=(72, 1)))
    layerArray.append(BatchNormalization())
    layerArray.append(MaxPooling1D(pool_size=3, strides=2, padding='same'))
    layerArray.append(Flatten())
    layerArray.append(Dense(64, activation='relu'))
    layerArray.append(Dense(3, activation='softmax'))
    return layerArray