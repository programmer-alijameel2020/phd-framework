# Evolutionary Programmed Neural Network Classifier (EvP-NNC)
# by: Ali J. Hashim

import keras
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Conv1D, BatchNormalization, MaxPooling1D
import matplotlib.pyplot as plt
from keras.callbacks import CSVLogger
from EvPNNC.EvaluationMetric import evaluationMetric
from EvPNNC.preprocessor import preprocessor
import seaborn as sns
from EvPNNC.NeuralNetwork import modelConstruction, initializeLayerArray, adaptiveLayer, initParameters, parameters
from keras_visualizer import visualizer

palette = sns.color_palette("rocket_r")


class EvPNNC_Class:
    def __init__(self):
        # get the layer information within current autoEncoder implementation
        layerArray = initializeLayerArray()
        # creates the autoEncoder classifier
        model = modelConstruction(layerArray)
        evaluationClass = evaluationMetric()
        model.compile(loss="categorical_crossentropy", optimizer="adam",
                      metrics=['accuracy', evaluationClass.f1_m, evaluationClass.precision_m, evaluationClass.recall_m,
                               evaluationClass.meanSquaredError])

        self.model = model
        self.evaluationClass = evaluationClass
        self.acc_history = []
        self.f1_history = []
        self.precision_history = []
        self.rec_history = []
        self.loss_history = []
        self.MSE_history = []

        self.number_of_classes = None
        self.y_test = None
        self.X_test = None
        self.y_train = None
        self.X_train = None
        self.model_iteration = None
        self.dataset = None
        self.epochs = None
        self.modelHistory = None

    def return_acc_history(self):
        return self.acc_history

    def get_layer_weight(self, i):
        return self.model.layers[i].get_weights()

    def set_layer_weight(self, i, weight):
        self.model.layers[i].set_weights(weight)

    def train(self):
        csv_logger = CSVLogger('metrics.csv', append=True)
        self.modelHistory = self.model.fit(self.X_train, self.y_train, batch_size=32, epochs=self.epochs, verbose=1,
                                           shuffle=True, callbacks=[csv_logger], validation_data=(
                self.X_test, self.y_test))  # , validation_data =(X_test, y_test)

    def test(self):
        loss, acc, f1_score, precision, recall, MSE = self.model.evaluate(self.X_test, self.y_test)
        self.acc_history.append(acc)
        self.f1_history.append(f1_score)
        self.precision_history.append(precision)
        self.rec_history.append(recall)
        self.loss_history.append(loss)
        self.MSE_history.append(MSE)
        return acc

    def load_layer_weights(self, weights):
        self.model.set_weights(weights)

    def give_weights(self):
        return self.model.get_weights()

    def weight_len(self):
        i = 0
        for j in self.model.layers:
            i += 1
        return i

    def architecture(self):
        self.model.summary()

    # parameterInitialization() imports the data preprocessor to extract train and testing
    def parameterInitialization(self, dataset, epochs):
        PreProcessorClass = preprocessor()
        number_of_classes = 8
        X_train, y_train, X_test, y_test = PreProcessorClass.data_preprocessor(dataset, number_of_classes)

        self.dataset = dataset
        self.epochs = epochs
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.number_of_classes = number_of_classes




















    def runModel(self):
        self.train()
        self.test()
        self.model.summary()
        history = self.modelHistory.history
        filename = 'layer_'
        # Open the file
        with open(filename + 'report.txt', 'w') as fh:
            # Pass the file handle in as a lambda function to make it callable
            self.model.summary(print_fn=lambda x: fh.write(x + '\n'))

        epochs = range(1, len(history['loss']) + 1)
        acc = history['accuracy']
        loss = history['loss']
        # val_acc = history['val_accuracy']
        # val_loss = history['val_loss']

        # visualize training and val accuracy
        """"
        plt.figure(figsize=(10, 5))
        plt.title('Training accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.plot(epochs, acc, label='accuracy')
        # plt.plot(epochs, val_acc, label='val_acc')
        plt.legend()
        plt.show()
        """

    def averageResultsCalculater(self, metricsData):
        metrics = pd.read_csv(metricsData)
        PreProcessorClass = preprocessor()
        print(PreProcessorClass.digitFloat(metrics['accuracy'].mean()))
        print(PreProcessorClass.digitFloat(metrics['val_accuracy'].mean()))
        print(PreProcessorClass.digitFloat(metrics['meanSquaredError'].mean()))

    def netAdaptation(self):
        # get the new structure
        newNetStructure = adaptiveLayer()
        # assign the model from the layer adaptation process
        self.model = modelConstruction(newNetStructure)
        # set the metrics and compilation
        self.model.compile(loss="categorical_crossentropy", optimizer="adam",
                           metrics=['accuracy', self.evaluationClass.f1_m, self.evaluationClass.precision_m,
                                    self.evaluationClass.recall_m, self.evaluationClass.meanSquaredError])

        # run the training and testing models
        self.train()
        self.test()
        self.model.summary()

    def netAdjustment(self):
        # get the current structure
        layerArray = initializeLayerArray()

        # get the current parameters
        parametersArray = initParameters()

        # adjust the parameters according to the adjustment factor
        parametersArray['filters'] = parametersArray['filters'] + 1
        parametersArray['kernel_size'] = parametersArray['kernel_size'] + 1
        parametersArray['padding'] = parametersArray['padding']
        parametersArray['pool_size'] = parametersArray['pool_size'] + 2
        parametersArray['strides'] = parametersArray['strides'] + 1
        parametersArray['unit_1'] = parametersArray['unit_1'] * 2
        parametersArray['unit_2'] = parametersArray['unit_2'] * 2
        parametersArray['unit_3'] = parametersArray['unit_3'] * 2
        parametersArray['unit_4'] = parametersArray['unit_4']
        parametersArray['activation'] = parametersArray['activation']

        # adjust the parameters according to the adjustment factor
        parameters['filters'] = parametersArray['filters'] + 1
        parameters['kernel_size'] = parametersArray['kernel_size'] + 1
        parameters['padding'] = parametersArray['padding']
        parameters['pool_size'] = parametersArray['pool_size'] + 2
        parameters['strides'] = parametersArray['strides'] + 1
        parameters['unit_1'] = parametersArray['unit_1'] * 2
        parameters['unit_2'] = parametersArray['unit_2'] * 2
        parameters['unit_3'] = parametersArray['unit_3'] * 2
        parameters['unit_4'] = parametersArray['unit_4']
        parameters['activation'] = parametersArray['activation']
        
        # deploy the model according to the new layers
        self.model = modelConstruction(layerArray)
        # set the metrics and compilation
        self.model.compile(loss="categorical_crossentropy", optimizer="adam",
                           metrics=['accuracy', self.evaluationClass.f1_m, self.evaluationClass.precision_m,
                                    self.evaluationClass.recall_m, self.evaluationClass.meanSquaredError])

        # run the training and testing models
        self.train()
        self.test()
        self.model.summary()