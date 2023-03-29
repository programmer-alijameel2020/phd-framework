# Evolutionary Programmed Neural Network Classifier (EvP-NNC)
# by: Ali J. Hashim

import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Conv1D, BatchNormalization, MaxPooling1D
import matplotlib.pyplot as plt
from keras.callbacks import CSVLogger
from EvPNNC.EvaluationMetric import evaluationMetric
from EvPNNC.preprocessor import preprocessor
import seaborn as sns
from EvPNNC.NeuralNetwork import modelConstruction, initializeLayerArray
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
                      metrics=['accuracy', evaluationClass.f1_m, evaluationClass.precision_m, evaluationClass.recall_m])

        self.model = model
        self.acc_history = []
        self.f1_history = []
        self.precision_history = []
        self.rec_history = []
        self.loss_history = []

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
                       shuffle=True, callbacks=[csv_logger], validation_data=(self.X_test, self.y_test))  # , validation_data =(X_test, y_test)
    def test(self):
        loss, acc, f1_score, precision, recall = self.model.evaluate(self.X_test, self.y_test)
        self.acc_history.append(acc)
        self.f1_history.append(f1_score)
        self.precision_history.append(precision)
        self.rec_history.append(recall)
        self.loss_history.append(loss)
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
        X_train, y_train, X_test, y_test = PreProcessorClass.data_preprocessor(dataset)
        number_of_classes = 10
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
        epochs = range(1, len(history['loss']) + 1)
        acc = history['accuracy']
        loss = history['loss']
        # val_acc = history['val_accuracy']
        # val_loss = history['val_loss']

        # visualize training and val accuracy
        plt.figure(figsize=(10, 5))
        plt.title('Training accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.plot(epochs, acc, label='accuracy')
        # plt.plot(epochs, val_acc, label='val_acc')
        plt.legend()
        plt.show()
