import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv1D, BatchNormalization, MaxPooling1D
import json

import keras

# Layer types array
layerTypes = ['Conv1D', 'BatchNormalization', 'MaxPooling1D', 'Flatten', 'Dense']

# the layers parameters with corresponding values
parameters = {
    'filters': 64,
    'kernel_size': 6,
    'padding': 'same',
    'input_shape': (72, 1),
    'pool_size': 3,
    'strides': 2,
    'unit_1': 140,
    'unit_2': 64,
    'unit_3': 32,
    'unit_4': 16,
    'unit_5': 8,
    'unit_6': 2,
    'unit_7': 1,
    'unit_out': 8,
    'activation': ['sigmoid', 'relu', 'softmax']
}


# Construct the model according to the passed Layer array
def modelConstruction(LayerArray):
    # design an autoEncoder neural network classifier for anomaly detection
    # LayerArray: is the parameter that contains the layer types and configurations
    model = Sequential()
    for layer in LayerArray:
        model.add(layer)
    return model


# returns the current network parameters
def initParameters():
    return parameters


# adjust the current parameters
def adjustParameters(newParameters):
    # adjust the parameters according to the adjustment factor
    return newParameters


# A special type function that takes the layer string as a string and convert it into a keras layer
def layerStringConverter(layerStringArray):
    layer_array = []
    layer = None
    for layerName in layerStringArray:
        if layerName == 'Conv1D':
            layer = Conv1D
        elif layerName == 'BatchNormalization':
            layer = BatchNormalization
        elif layerName == 'MaxPooling1D':
            layer = MaxPooling1D
        elif layerName == 'Flatten':
            layer = Flatten
        elif layerName == 'Dense':
            layer = Dense
        layer_array.append(layer)
    return layer_array


# A Spacial type function to create a json array structure to be stored for the current keras model. The parameters
# are assigned according to the layer type
def createJsonObject(layerTypes, parameters):
    layer_array = []

    for layer in layerTypes:
        if layer == 'Conv1D':
            layerParameters = {
                "filters": parameters['filters'],
                "kernel_size": parameters['kernel_size'],
                "activation": parameters['activation'][1],
                "padding": parameters['padding'],
                "input_shape": parameters['input_shape']
            }
        elif layer == 'BatchNormalization':
            layerParameters = {}

        elif layer == 'MaxPooling1D':
            layerParameters = {
                "pool_size": parameters['pool_size'],
                "strides": parameters['strides'],
                "padding": parameters['padding']
            }

        elif layer == 'Flatten':
            layerParameters = {}

        elif layer == 'Dense':
            layerParameters = {"unit": parameters['unit_1'], "activation": parameters['activation'][0]}

        # Build the json object and append into the layer array
        jsonObject = {
            "layer_type": layer,
            "parameters": layerParameters
        }
        layer_array.append(jsonObject)

    # Store the network structure into .json file for evaluation purposes
    with open('structure.json', 'w') as outfile:
        json.dump(layer_array, outfile)


# The main layer initialization function
def initializeLayerArray():
    # adding the processing layers to the layer array to be forwarded for construction
    # Initialize the parameters of the processing layers

    # Converts the string array into a keras model using the special type function
    layers = layerStringConverter(layerTypes)

    layer_count = 1
    # Stack the processing layers
    ConvLayer = layers[0](filters=parameters['filters'], kernel_size=parameters['kernel_size'],
                          activation=parameters['activation'][1], padding=parameters['padding'],
                          input_shape=parameters['input_shape'])
    BatchNormalizationLayer = layers[1]()
    MXPoolingLayer = layers[2](pool_size=parameters['pool_size'], strides=parameters['strides'],
                               padding=parameters['padding'])
    FlattenLayer = layers[3]()

    Dense1 = layers[4](parameters['unit_1'], activation=parameters['activation'][0])
    Dense2 = layers[4](parameters['unit_2'], activation=parameters['activation'][0])
    Dense3 = layers[4](parameters['unit_3'], activation=parameters['activation'][0])
    Dense4 = layers[4](parameters['unit_4'], activation=parameters['activation'][0])
    Dense_Encoder_Out = layers[4](parameters['unit_out'], activation=parameters['activation'][0])
    Dense_out = layers[4](parameters['unit_out'], activation=parameters['activation'][2])
    Dense_Decoder_Out = layers[4](parameters['unit_1'], activation=parameters['activation'][2])

    encoder = [Dense1, Dense2, Dense3, Dense4, Dense_Encoder_Out]
    decoder = [Dense4, Dense3, Dense2, Dense_Decoder_Out]

    # create a conditional statement to assign the parameters according to the layer type and store in layer array
    createJsonObject(layerTypes, parameters)
    return encoder, decoder


def adaptiveLayer():
    # Get the current neural network structure
    currentNetworkStructure = initializeLayerArray()
    # Identify the layer insertion point
    InsertionPoint = 4
    layers = layerStringConverter(layerTypes)
    DensLayer = layers[4](parameters['unit_2'], activation=parameters['activation'][0])
    currentNetworkStructure.insert(InsertionPoint, DensLayer)
    return currentNetworkStructure
