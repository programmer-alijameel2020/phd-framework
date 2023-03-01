import keras
import pandas as pd
import numpy as np
from keras.callbacks_v1 import TensorBoard
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.callbacks import EarlyStopping, CSVLogger
import seaborn as sns
from Framework.EvaluationMetric import evaluationMetric

from tensorflow import keras
from keras import Model, Sequential, layers
from keras.layers import Dense, Conv1D, BatchNormalization, MaxPooling1D, Flatten

from Framework.Preprocessor import preprocessor

palette = sns.color_palette("rocket_r")


class Autoencoder(Model):
    def __init__(self):
        super(Autoencoder, self).__init__()
        """
            Build the model structure. The model structure consists of encoder model and decoder model where the 
            encoder moder responsible for encoding the signal into simplified representation and the decoder moder 
            responsible for the reconstruction of the decoder signal 
        """
        enc_model = Sequential()
        # encoder
        enc_model.add(Conv1D(filters=64, kernel_size=6, activation='relu',
                         padding='same', input_shape=(72, 1)))

        enc_model.add(Dense(140, activation='sigmoid'))
        enc_model.add(Dense(64, activation='relu'))
        enc_model.add(Dense(32, activation='relu'))
        enc_model.add(Dense(16, activation='relu'))
        enc_model.add(Dense(8, activation='relu'))
        self.encoder = enc_model
        # decoder
        dec_model = Sequential()
        dec_model.add(Dense(8, activation='relu'))
        dec_model.add(Dense(16, activation='relu'))
        dec_model.add(Dense(32, activation='relu'))
        dec_model.add(Dense(64, activation='relu'))
        dec_model.add(Dense(140, activation='sigmoid'))
        dec_model.add(Conv1D(filters=64, kernel_size=6, activation='relu',
                         padding='same', input_shape=(72, 1)))
        self.decoder = dec_model
        # Performance metrics
        self.acc_history = []
        self.f1_history = []
        self.precision_history = []
        self.rec_history = []
        self.loss_history = []

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def return_acc_history(self):
        return self.acc_history

    def get_layer_weight(self, i):
        return self.encoder.layers[i].get_weights()

    def set_layer_weight(self, i, weight):
        self.encoder.layers[i].set_weights(weight)

    def load_layer_weights(self, weights):
        self.encoder.set_weights(weights)

    def give_weights(self):
        return self.encoder.get_weights()

    def weight_len(self):
        i = 0
        for j in self.encoder.layers:
            i += 1
        return i

    def architecture(self):
        self.encoder.summary()

    def run_autoEncoder(self, dataset, epochs, batch_size, stopping_patience, generation):
        # Initialize the dataset info
        # Perform preprocessor to extract train and testing
        PreProcessorClass = preprocessor()
        X_train, y_train, X_test, y_test = PreProcessorClass.data_preprocessor(dataset)
        print("shaped data:", X_train.shape)
        autoencoder = Autoencoder()
        evaluationClass = evaluationMetric()
        autoencoder.compile(loss="binary_crossentropy", optimizer="adam")


        autoencoder.fit(X_train, X_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        shuffle=True,
                        validation_data=(X_test, X_test),
                        callbacks=[TensorBoard(log_dir='/autoencoder')])



class EvolutionaryAutoEncoder:
    def __init__(self, model_iteration, population_size, mutation_rate, dataset, epochs, batch_size,
                 stopping_patience=2, generations=50):
        self.norm_acc = None
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.population = None
        self.children_population_weights = []
        self.acces = []
        self.norm_acces = []
        self.model_iteration = model_iteration
        self.dataset = dataset
        self.epochs = epochs
        self.batch_size = batch_size
        self.stopping_patience = stopping_patience

    def create_population(self):
        self.population = [Autoencoder() for i in
                           range(self.population_size)]

    def runEncoderEvolution(self, generation):
        for member in self.population:
            member.run_autoEncoder(self.dataset, self.epochs, self.batch_size, self.stopping_patience, generation)

    def normalize(self):
        sum_ = sum(self.acc)
        self.norm_acc = [i / sum_ for i in self.acc]
        print("\nNormalization sum: ", sum(self.norm_acc))
        # assert sum(self.norm_acc) == 1

    def clear_losses(self):
        self.norm_acc = []
        self.acc = []

    def mutate(self):
        for member in self.population:
            for i in range(member.weight_len()):
                if np.random.random() < self.mutation_rate:
                    print("\nMutation!")
                    old_weight = member.get_layer_weight(i)
                    new_weight = [np.random.uniform(low=-1, high=1, size=old_weight[i].shape) for i in
                                  range(len(old_weight))]
                    member.set_layer_weight(i, new_weight)

    def reproduction(self):
        """
        Reproduction through midpoint crossover method
        """
        population_idx = [i for i in range(len(self.population))]
        for i in range(len(self.population)):
            # selects two parents probabilistic accroding to the fitness
            if sum(self.norm_acc) != 0:
                parent1 = np.random.choice(population_idx, p=self.norm_acc)
                parent2 = np.random.choice(population_idx, p=self.norm_acc)
            else:
                # if there are no "best" parents choose randomly
                parent1 = np.random.choice(population_idx)
                parent2 = np.random.choice(population_idx)

            # picking random midpoint for crossing over name/DNA
            parent1_weights = self.population[parent1].give_weights()
            parent2_weights = self.population[parent2].give_weights()

            mid_point = np.random.choice([i for i in range(len(parent1_weights))])
            # adding DNA-Sequences of the parents to final DNA
            self.children_population_weights.append(parent1_weights[:mid_point] + parent2_weights[mid_point:])
        # old population gets the new and proper weights
        for i in range(len(self.population)):
            for j in range(len(self.children_population_weights)):
                self.population[i].load_layer_weights(self.children_population_weights[j])

    def run_evolution(self):
        for episode in range(self.generations):
            self.clear_losses()
            self.runEncoderEvolution(episode)
            if episode != self.generations - 1:
                self.normalize()
                self.reproduction()
                self.mutate()
            else:
                pass

        # plotting history:
        for a in range(self.generations):
            plt.plot(label='accuracy for gen: ' + str(a))
            for member in self.population:
                plt.plot(member.acc_history)
        plt.xlabel("Generations")
        plt.ylabel("Accuracy")
        plt.show()

        for a in range(self.generations):
            for member in self.population:
                plt.plot(member.loss_history)
        plt.xlabel("Generations")
        plt.ylabel("Loss")
        plt.show()
