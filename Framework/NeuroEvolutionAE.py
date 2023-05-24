import matplotlib
import pandas as pd
import numpy as np
from keras.utils import plot_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.callbacks import EarlyStopping, CSVLogger
import seaborn as sns
from keras.models import Sequential, Model
from keras.layers import Dense

from Framework.EvaluationMetric import evaluationMetric

palette = sns.color_palette("rocket_r")


def get_clf_eval(y_test, pred=None, pred_proba=None):
    confusion = confusion_matrix(y_test, pred)
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)

    roc_auc = roc_auc_score(y_test, pred_proba)
    print('confusion matrix')
    print(confusion)

    # ROC-AUC print
    print('accuracy: {0:.4f}, precision: {1:.4f}, recall: {2:.4f},\
    F1: {3:.4f}, AUC:{4:.4f}'.format(accuracy, precision, recall, f1, roc_auc))
    return confusion


def predict(model, data, threshold):
    reconstructions = model(data)
    loss = tf.keras.losses.mae(reconstructions, data)
    return tf.math.less(loss, threshold)


def print_stats(predictions, labels):
    print("Accuracy = {}".format(accuracy_score(labels, predictions)))
    print("Precision = {}".format(precision_score(labels, predictions)))
    print("Recall = {}".format(recall_score(labels, predictions)))


class Autoencoder(Model):
    def __init__(self):
        super(Autoencoder, self).__init__()
        """
        Build the model structure. The model structure consists of encoder model and decoder model where the 
        encoder moder responsible for encoding the signal into simplified representation and the decoder moder 
        responsible for the reconstruction of the decoder signal 
        """
        encoder_model = Sequential()
        encoder_model.add(Dense(140, activation='sigmoid', input_shape=(140, 140)))
        encoder_model.add(Dense(64, activation='relu'))
        encoder_model.add(Dense(32, activation='relu'))
        encoder_model.add(Dense(16, activation='relu'))
        encoder_model.add(Dense(8, activation='relu'))

        self.encoder = encoder_model

        decoder_model = Sequential()
        decoder_model.add(Dense(8, activation='relu'))
        decoder_model.add(Dense(16, activation='relu'))
        decoder_model.add(Dense(32, activation='relu'))
        decoder_model.add(Dense(64, activation='relu'))
        decoder_model.add(Dense(140, activation='sigmoid', input_shape=(140, 140)))

        self.decoder = decoder_model
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

    def test(self):
        loss, acc, f1_score, precision, recall = self.encoder.evaluate(self.X_test, self.y_test)
        self.acc_history.append(acc)
        self.f1_history.append(f1_score)
        self.precision_history.append(precision)
        self.rec_history.append(recall)
        self.loss_history.append(loss)
        return acc

    def run_autoEncoder(self, dataset, epochs, batch_size, stopping_patience, generation):
        df = pd.read_csv(dataset, sep='  ', header=None, engine='python')
        print(df.shape)
        # df = pd.read_csv('storage/dataset/test.txt', sep='  ', header=None, engine='python')
        # print(df.head())
        df.columns
        # print(df.describe())
        # splitting into train test data
        train_data, test_data, train_labels, test_labels = train_test_split(df.values, df.values[:, 0:1], test_size=0.2,
                                                                            random_state=111)
        # Initializing a MinMax Scaler
        scaler = MinMaxScaler()

        # Fitting the train data to the scaler
        data_scaled = scaler.fit(train_data)

        # Scaling dataset according to weights of train data
        train_data_scaled = data_scaled.transform(train_data)
        test_data_scaled = data_scaled.transform(test_data)
        train_data.shape
        # Making pandas dataframe for the normal and anomaly train data points
        normal_train_data = pd.DataFrame(train_data_scaled).add_prefix('c').query('c0 == 0').values[:, 1:]
        anomaly_train_data = pd.DataFrame(train_data_scaled).add_prefix('c').query('c0 > 0').values[:, 1:]

        # print("Anomaly training data: ", anomaly_train_data)

        # Making pandas dataframe for the normal and anomaly test data points
        normal_test_data = pd.DataFrame(test_data_scaled).add_prefix('c').query('c0 == 0').values[:, 1:]
        anomaly_test_data = pd.DataFrame(test_data_scaled).add_prefix('c').query('c0 > 0').values[:, 1:]

        # Instantiating the Autoencoder
        model = Autoencoder()

        # creating an early_stopping
        early_stopping = EarlyStopping(monitor='val_loss',
                                       patience=stopping_patience,
                                       mode='min')

        csv_logger = CSVLogger('results/metrics_' + str(generation) + '.csv', append=True)

        evaluationClass = evaluationMetric()
        # Compiling the model
        model.compile(optimizer='adam',
                      loss='mae', metrics=['accuracy', evaluationClass.f1_m, evaluationClass.precision_m,
                                           evaluationClass.recall_m])
        # Training the model
        history = model.fit(normal_train_data, normal_train_data,
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_data=(train_data_scaled[:, 1:], train_data_scaled[:, 1:]),
                            shuffle=True,
                            callbacks=[early_stopping, csv_logger])

        # predictions for normal test data points
        encoder_out = model.encoder(normal_test_data).numpy()
        decoder_out = model.decoder(encoder_out).numpy()

        # print("Test data:", encoder_out)
        # print("Predictions:", decoder_out)
        # print(decoder_out.shape)
        # print(normal_test_data[0], 'b')
        # print(decoder_out[0], 'r')

        # predictions for anomaly test data points
        encoder_out_a = model.encoder(anomaly_test_data).numpy()
        decoder_out_a = model.decoder(encoder_out_a).numpy()

        # print("Anomaly training data", anomaly_train_data[0], 'b')
        # print("Decoder out: ", decoder_out_a[0], 'r')

        # reconstruction loss for normal test data
        reconstructions = model.predict(normal_test_data)
        train_loss = tf.keras.losses.mae(reconstructions, normal_test_data)

        # Plotting histogram for recontruction loss for normal test data
        # print("Training loss", train_loss)
        # print("Mean: ", np.mean(train_loss))
        # print("Standard deviation: ", np.std(train_loss))

        # reconstruction loss for anomaly test data
        reconstructions_a = model.predict(anomaly_test_data)
        train_loss_a = tf.keras.losses.mae(reconstructions_a, anomaly_test_data)

        # Plotting histogram for reconstruction loss for anomaly test data
        # print("Reconstructed training loss: ", train_loss_a)
        # print("Mean: ", np.mean(train_loss_a))
        # print("Standard deviation: ", np.std(train_loss_a))

        # setting threshold
        threshold = np.mean(train_loss) + 2 * np.std(train_loss)
        # print("Threshold: ", threshold)

        # print("normal_test_data: ", normal_test_data)
        # print("Predictions: ", reconstructions)
        if (generation >= 0):
            # Plotting the normal and anomaly losses with the threshold
            plt.hist(train_loss, bins=50, density=True, label="Normal (train data loss)", alpha=.6, color="green")
            plt.hist(train_loss_a, bins=50, density=True, label="Anomaly (test data loss)", alpha=.6, color="red")
            plt.axvline(threshold, color='#000000', linewidth=2, linestyle='dashed',
                        label='Threshold value: {:0.3f}'.format(threshold))
            plt.legend(loc='upper right')
            plt.title("Abnormality detection report for (" + str(epochs) + ") epochs with generation: (" + str(
                generation) + ")")
            plt.show()

        # Number of correct predictions for Normal test data
        preds = tf.math.less(train_loss, threshold)

        print("Number of correct predictions: ", tf.math.count_nonzero(preds))
        # Number of correct predictions for Anomaly test data
        preds_a = tf.math.greater(train_loss_a, threshold)
        print("Number of correct predictions for anomaly data: ", tf.math.count_nonzero(preds_a))
        # print(preds_a.shape)
        # print(reconstructions_a)
        if (generation >= 0):
            plt.plot(reconstructions_a[0], label="predictions for abnormality", alpha=.6,
                     marker=matplotlib.markers.CARETUPBASE, color="black")
            plt.plot(anomaly_test_data[0], label="Anomaly test data", alpha=.6, color="red", marker="s")
            plt.legend(loc='upper left')
            plt.fill_between(np.arange(140), decoder_out_a[0], anomaly_test_data[0], color='#FFCDD2')
            # plt.plot(reconstructions_a[0], label="predictions for anomaly data", marker=matplotlib.markers.CARETUPBASE)
            plt.title("Prediction signal of the AEVAE for (" + str(epochs) + ") epochs with generation (" + str(
                generation) + ")")
            plt.show()

        """
        plt.hist(reconstructions_a[0], bins=50, density=True, label="Predictions for anomaly data", alpha=.6, color="green")
        plt.hist(anomaly_test_data[0], bins=50, density=True, label="Anomaly test data", alpha=.6, color="red")
        # plt.plot(reconstructions_a[0], label="predictions for anomaly data", marker=matplotlib.markers.CARETUPBASE)
        plt.title("Abnormality detection report for (" + str(epochs) + ") epochs")
        plt.show()
        """
        preds = predict(model, test_data, threshold)
        print_stats(preds, test_labels)
        confusion_matrix = get_clf_eval(test_labels, preds, preds)
        plt.figure(figsize=(8, 6))
        sns.set(font_scale=2)
        sns.set_style("white")
        sns.heatmap(confusion_matrix, cmap='gist_yarg_r', annot=True, fmt='d')
        plt.show()

        """
        # Build the models
        self.encoder.build(input_shape=(None, 140))
        self.decoder.build(input_shape=(None, 8))

        self.encoder.summary()
        self.decoder.summary()
        """

        # plt.hist(preds_a, bins=50, density=True, label="predictions for anomalies", alpha=.6, color="red")

        # Plotting the normal and anomaly losses with the threshold

        # plt.plot(preds, label="predictions for normal data", marker=matplotlib.markers.CARETUPBASE)
        # plt.plot(preds_a, label="predictions for anomalies",)


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

    """
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

    """
