# NOTE: All the neuroEvolution CNN programming are placed in the same class
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Conv1D, BatchNormalization, MaxPooling1D
import matplotlib.pyplot as plt
from keras.callbacks import CSVLogger
from Framework.EvaluationMetric import evaluationMetric
from Framework.Preprocessor import preprocessor
import seaborn as sns

palette = sns.color_palette("rocket_r")


class Network:
    def __init__(self, dataset, model_iteration):
        # Initialize the dataset info
        # Perform preprocessor to extract train and testing
        PreProcessorClass = preprocessor()
        X_train, y_train, X_test, y_test = PreProcessorClass.data_preprocessor(dataset)
        number_of_classes = 10

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.number_of_classes = number_of_classes

        model = Sequential()
        model.add(Conv1D(filters=64, kernel_size=6, activation='relu',
                         padding='same', input_shape=(72, 1)))
        model.add(BatchNormalization())
        # adding a pooling layer
        model.add(MaxPooling1D(pool_size=(3), strides=2, padding='same'))

        model.add(Conv1D(filters=64, kernel_size=6, activation='relu',
                         padding='same', input_shape=(72, 1)))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=(3), strides=2, padding='same'))

        model.add(Conv1D(filters=64, kernel_size=6, activation='relu',
                         padding='same', input_shape=(72, 1)))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=(3), strides=2, padding='same'))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(3, activation='softmax'))

        evaluationClass = evaluationMetric()
        model.compile(loss="categorical_crossentropy", optimizer="adam",
                      metrics=['accuracy', evaluationClass.f1_m, evaluationClass.precision_m, evaluationClass.recall_m])
        self.model = model
        self.acc_history = []
        self.f1_history = []
        self.precision_history = []
        self.rec_history = []
        self.loss_history = []
        self.model_iteration = model_iteration

    def return_acc_history(self):
        return self.acc_history

    def get_layer_weight(self, i):
        return self.model.layers[i].get_weights()

    def set_layer_weight(self, i, weight):
        self.model.layers[i].set_weights(weight)

    def train(self):
        csv_logger = CSVLogger('metrics.csv', append=True)
        self.model.fit(self.X_train, self.y_train, batch_size=32, epochs=self.model_iteration, verbose=1,
                       shuffle=True, callbacks=[csv_logger])  # , validation_data =(X_test, y_test)

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


class EvolutionaryCNN:
    def __init__(self, model_iteration, population_size, mutation_rate, dataset_path, generations=50):
        self.norm_acc = None
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.population = None
        self.children_population_weights = []
        self.acces = []
        self.norm_acces = []
        self.dataset = dataset_path
        self.model_iteration = model_iteration

    def create_population(self):
        self.population = [Network(dataset=self.dataset, model_iteration = self.model_iteration) for i in range(self.population_size)]

    def train_generation(self):
        for member in self.population:
            member.train()

    def predict(self):
        for member in self.population:
            acc = member.test()
            self.acc.append(acc)
            # logging.info("Losses: {}".format(loss))

    def normalize(self):
        sum_ = sum(self.acc)
        self.norm_acc = [i / sum_ for i in self.acc]
        print("\nNormalization sum: ", sum(self.norm_acc))
        # assert sum(self.norm_acc) == 1

    def show_weights(self):
        for i in parent_weights:
            print(i)

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
            self.train_generation()
            self.predict()
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
        """""
        for a in range(self.generations):
            for member in self.population:
                plt.plot(member.f1_history)
        plt.xlabel("Generations")
        plt.ylabel("F1 value")
        plt.show()

        for a in range(self.generations):
            for member in self.population:
                plt.plot(member.precision_history)
        plt.xlabel("Generations")
        plt.ylabel("Precision")
        plt.show()

        for a in range(self.generations):
            for member in self.population:
                plt.plot(member.rec_history)
        plt.xlabel("Generations")
        plt.ylabel("Recall")
        plt.show()
        """
