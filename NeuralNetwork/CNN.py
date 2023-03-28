# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


# A programmatic implementation for the PHD project: A NeuroEvolution Network for Anomaly Detection in Computer Network
# The system programming is distributed over multiple classes where each class provides a particular service

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os, re, time, math, tqdm, itertools
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.offline as pyo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
import keras
from keras.layers import Conv2D, Conv1D, MaxPooling2D, MaxPooling1D, Flatten, BatchNormalization, Dense
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.callbacks import CSVLogger, ModelCheckpoint
from sklearn.utils import resample

from EvaluationMetric import evaluationMetric
from NeuralNetworkClass import convNeuralNetwork
from EvolutionaryNeuralNetwork.GeneticNeuralNetwork import GeneticAlgorithm
from preprocessor import preprocessor


# Press the green button in the gutter to run the program.

def run_network_model():
    dataset_path = 'storage/dataset/02-14-2018.csv'
    """""
    preprocessor(): a class that responsible for data cleaning and organizing 
    data_preprocessor(path): a function that takes the dataset path as input and produce the training, testing set as
        output 
    """
    PreProcessorClass = preprocessor()
    X_train, y_train, X_test, y_test = PreProcessorClass.data_preprocessor(dataset_path)
    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)
    # Call a class for Convolutional Neural Network

    model = convNeuralNetwork(x_train_param=X_train, y_train_param=y_train, x_test_param=X_test, y_test_param=y_test)
    modelFunction = model.model_function()
    modelFunction.summary()
    logger = CSVLogger('logs.csv', append=True)

    his = modelFunction.fit(X_train, y_train, epochs=15, batch_size=32,
                    validation_data=(X_test, y_test), callbacks=[logger])

    # Visualization of Results (CNN)
    ' lets make a graphical visualization of results obtained by applying CNN to our data.'
    # check the model performance on test data
    scores = modelFunction.evaluate(X_test, y_test)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

    # check history of model
    history = his.history
    history.keys()

    print(history)

    epochs = range(1, len(history['loss']) + 1)
    acc = history['accuracy']
    loss = history['loss']
    val_acc = history['val_accuracy']
    val_loss = history['val_loss']

    # visualize training and val accuracy
    plt.figure(figsize=(10, 5))
    plt.title('Training and Validation Accuracy (CNN)')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.plot(epochs, acc, label='accuracy')
    plt.plot(epochs, val_acc, label='val_acc')
    plt.legend()
    plt.show()

    # visualize train and val loss
    plt.figure(figsize=(10, 5))
    plt.title('Training and Validation Loss(CNN)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(epochs, loss, label='loss', color='g')
    plt.plot(epochs, val_loss, label='val_loss', color='r')
    plt.legend()
    plt.show()

    GA = GeneticAlgorithm(population_size=5, mutation_rate=0.05, generations=5)
    GA.create_population()
    GA.run_evolution()
