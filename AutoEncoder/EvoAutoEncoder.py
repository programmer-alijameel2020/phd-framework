import matplotlib
import pandas as pd
import numpy as np
import umap
from keras.utils import plot_model
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.callbacks import EarlyStopping, CSVLogger, TensorBoard
import seaborn as sns
from keras.models import Sequential, Model
from keras.layers import Dense
from keras.utils import plot_model

from AutoEncoder.NeuralNetwork import initializeLayerArray, modelConstruction
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
        # get the layer information within current autoEncoder implementation

        """"
        encoder_model = Sequential()
        encoder_model.add(Dense(140, activation='relu'))
        encoder_model.add(Dense(64, activation='relu'))
        encoder_model.add(Dense(32, activation='relu'))
        encoder_model.add(Dense(16, activation='relu'))
        encoder_model.add(Dense(8, activation='relu'))

        decoder_model = Sequential()
        decoder_model.add(Dense(16, activation='relu'))
        decoder_model.add(Dense(32, activation='relu'))
        decoder_model.add(Dense(64, activation='relu'))
        decoder_model.add(Dense(140, activation='sigmoid'))
        """

        # General hyperparameters
        self.parameters = {
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

        self.encoderLayerStack = []
        self.decoderLayerStack = []
        self.adaptiveLayerStatus = None

        # Set the layer stack initialization and Build the autoEncoder model
        encoder_model, decoder_model = self.layerStackInit()

        self.model = None
        self.encoder = encoder_model
        self.decoder = decoder_model

        # Performance metrics
        self.acc_history = []
        self.f1_history = []
        self.precision_history = []
        self.rec_history = []
        self.loss_history = []
        self.stopping_patience = None
        self.modelPool = []
        self.generations = None
        self.autoencoder = None


    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def layerStackInit(self):
        encoder_model = Sequential()
        encoder_model.add(Dense(140, activation='relu'))
        encoder_model.add(Dense(64, activation='relu'))
        encoder_model.add(Dense(32, activation='relu'))
        encoder_model.add(Dense(16, activation='relu'))
        encoder_model.add(Dense(8, activation='relu'))

        decoder_model = Sequential()
        decoder_model.add(Dense(16, activation='relu'))
        decoder_model.add(Dense(32, activation='relu'))
        decoder_model.add(Dense(64, activation='relu'))
        decoder_model.add(Dense(140, activation='sigmoid'))

        print("Adaption state:", self.adaptiveLayerStatus)

        if self.adaptiveLayerStatus is True:
            modifiedEncoder, modifiedDecoder = self.runModifications(encoder_model, decoder_model)
            return modifiedEncoder, modifiedDecoder
        else:
            return encoder_model, decoder_model



    def return_acc_history(self):
        return self.acc_history

    def get_layer_weight(self, i):
        return self.encoder.layers[i].get_weights(), self.decoder.layers[i].get_weights()

    def set_layer_weight(self, i, weight):
        self.encoder.layers[i].set_weights(weight)
        self.decoder.layers[i].set_weights(weight)

    def load_layer_weights(self, weights):
        self.encoder.set_weights(weights)
        self.decoder.set_weights(weights)

    def give_weights(self):
        return self.encoder.get_weights(), self.decoder.get_weights()

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
        sp = stopping_patience
        dataframe = dataset
        raw_data = dataframe.values
        dataframe.head().style.set_properties(**{'background-color': 'black',
                                                 'color': 'white',
                                                 'border-color': 'white'})
        self.generations = generation
        self.stopping_patience = stopping_patience
        colors = ['gold', 'mediumturquoise']
        labels = ['Normal', 'Abnormal']
        values = dataframe[140].value_counts() / dataframe[140].shape[0]

        # The last element contains the labels
        labels = raw_data[:, -1]

        # The other data points are the electrocadriogram data
        data = raw_data[:, 0:-1]

        train_data, test_data, train_labels, test_labels = train_test_split(
            data, labels, test_size=0.2, random_state=21
        )

        min_val = tf.reduce_min(train_data)
        max_val = tf.reduce_max(train_data)

        train_data = (train_data - min_val) / (max_val - min_val)
        test_data = (test_data - min_val) / (max_val - min_val)

        train_data = tf.cast(train_data, tf.float32)
        test_data = tf.cast(test_data, tf.float32)

        train_labels = train_labels.astype(bool)
        test_labels = test_labels.astype(bool)

        print("the length of train :", len(train_labels))
        print("the length of test :", len(test_labels))

        normal_train_data = train_data[train_labels]
        normal_test_data = test_data[test_labels]

        anomalous_train_data = train_data[~train_labels]
        anomalous_test_data = test_data[~test_labels]

        """
        plt.figure(figsize=(10, 8))
        sns.set(font_scale=2)
        sns.set_style("white")
        plt.grid()
        plt.plot(np.arange(140), normal_train_data[0], color='black', linewidth=3.0)
        plt.title("A Normal ECG")

        plt.figure(figsize=(10, 8))
        sns.set(font_scale=2)
        sns.set_style("white")
        plt.grid()
        plt.plot(np.arange(140), anomalous_train_data[0], color='red', linewidth=3.0)
        plt.title("An Anomalous ECG")
        plt.show()
        
        """
        # Start autoEncoder programming

        self.autoencoder = Autoencoder()

        # Compiling the model
        self.autoencoder.compile(optimizer='adam', loss='mae')

        # creating an early_stopping
        early_stopping = EarlyStopping(monitor='val_loss',
                                       patience=stopping_patience,
                                       mode='min')

        csv_logger = CSVLogger('results/metrics_' + str(generation) + '.csv', append=True)

        tensorboard_callback = TensorBoard(log_dir='logs')

        history = self.autoencoder.fit(normal_train_data, normal_train_data,
                                       epochs=epochs,
                                       batch_size=batch_size,
                                       validation_data=(test_data, test_data),
                                       shuffle=True,
                                       callbacks=[early_stopping, csv_logger, tensorboard_callback])




        """
        plt.figure(figsize=(10, 8))
        sns.set(font_scale=2)
        sns.set_style("white")
        plt.plot(history.history["loss"], label="Training Loss", linewidth=3.0)
        plt.plot(history.history["val_loss"], label="Validation Loss", linewidth=3.0)
        plt.legend()
        """

        encoded_imgs = self.autoencoder.encoder(normal_test_data).numpy()
        decoded_imgs = self.autoencoder.decoder(encoded_imgs).numpy()

        """
        plt.figure(figsize=(10, 8))
        sns.set(font_scale=2)
        sns.set_style("white")
        plt.plot(normal_test_data[0], 'black', linewidth=2)
        plt.plot(decoded_imgs[0], 'red', linewidth=2)
        plt.fill_between(np.arange(140), decoded_imgs[0], normal_test_data[0], color='lightcoral')
        plt.legend(labels=["Input", "Reconstruction", "Error"])
        plt.show()
        """

        encoded_imgs_normal = pd.DataFrame(encoded_imgs)
        encoded_imgs_normal['label'] = 1

        encoded_imgs = self.autoencoder.encoder(anomalous_test_data).numpy()
        decoded_imgs = self.autoencoder.decoder(encoded_imgs).numpy()

        """
        plt.figure(figsize=(10, 8))
        sns.set(font_scale=2)
        sns.set_style("white")
        plt.plot(anomalous_test_data[0], 'black', linewidth=2)
        plt.plot(decoded_imgs[0], 'red', linewidth=2)
        plt.fill_between(np.arange(140), decoded_imgs[0], anomalous_test_data[0], color='lightcoral')
        plt.legend(labels=["Input", "Reconstruction", "Error"])
        plt.show()
        """

        encoded_imgs_abnormal = pd.DataFrame(encoded_imgs)
        encoded_imgs_abnormal['label'] = 0

        all_encoded = pd.concat([encoded_imgs_normal, encoded_imgs_abnormal])
        mapper = umap.UMAP().fit(all_encoded.iloc[:, :8])

        reconstructions = self.autoencoder.predict(normal_train_data)
        train_loss = tf.keras.losses.mae(reconstructions, normal_train_data)
        np.mean(train_loss)

        # umap.plot.points(mapper, labels=all_encoded.iloc[:,8], theme='fire')

        # umap.plot.connectivity(mapper, show_points=True)

        plt.figure(figsize=(10, 8))
        sns.set(font_scale=2)
        sns.set_style("white")
        sns.histplot(train_loss, bins=50, kde=True, color='grey', linewidth=3)
        plt.axvline(x=np.mean(train_loss), color='g', linestyle='--', linewidth=3)
        plt.text(np.mean(train_loss), 200, "Mean", horizontalalignment='left',
                 size='small', color='black', weight='semibold')
        plt.xlabel("Train loss")
        plt.ylabel("No of examples")
        sns.despine()
        plt.show()

        threshold = np.mean(train_loss) + np.std(train_loss)
        print("Threshold: ", threshold)

        plt.figure(figsize=(12, 8))
        sns.set(font_scale=2)
        sns.set_style("white")
        sns.histplot(train_loss, bins=50, kde=True, color='grey', linewidth=3)
        plt.axvline(x=np.mean(train_loss), color='g', linestyle='--', linewidth=3)
        plt.text(np.mean(train_loss), 200, "Normal Mean", horizontalalignment='center',
                 size='small', color='black', weight='semibold')
        plt.axvline(x=threshold, color='b', linestyle='--', linewidth=3)
        plt.text(threshold, 250, "Threshold", horizontalalignment='center',
                 size='small', color='Blue', weight='semibold')
        plt.xlabel("Train loss")
        plt.ylabel("No of examples")
        sns.despine()
        plt.show()

        reconstructions = self.autoencoder.predict(anomalous_test_data)
        test_loss = tf.keras.losses.mae(reconstructions, anomalous_test_data)

        # Training anomalous
        plt.plot(reconstructions[0], label="predictions for abnormality in the training phase", alpha=.6,
                 marker=matplotlib.markers.CARETUPBASE, color="black")
        plt.plot(anomalous_train_data[0], label="Reconstruction test data", alpha=.6, color="red", marker="s")
        plt.legend(loc='best')
        plt.fill_between(np.arange(140), decoded_imgs[0], anomalous_train_data[0], color='#FFCDD2')
        # plt.plot(reconstructions_a[0], label="predictions for anomaly data", marker=matplotlib.markers.CARETUPBASE)
        plt.title("Prediction signal of the AEVAE for (" + str(epochs) + ") epochs with generation (" + str(
            generation) + ")")
        plt.legend(labels=["Input", "Reconstruction", "Error"])
        plt.show()

        # Error Between
        plt.plot(reconstructions[0], label="predictions for abnormality in the testing phase", alpha=.6,
                 marker=matplotlib.markers.CARETUPBASE, color="black")
        plt.plot(anomalous_test_data[0], label="Reconstruction test data", alpha=.6, color="red", marker="s")
        plt.legend(loc='best')
        plt.fill_between(np.arange(140), decoded_imgs[0], anomalous_test_data[0], color='#FFCDD2')
        # plt.plot(reconstructions_a[0], label="predictions for anomaly data", marker=matplotlib.markers.CARETUPBASE)
        plt.title("Prediction signal of the AEVAE for (" + str(epochs) + ") epochs with generation (" + str(
            generation) + ")")
        plt.legend(labels=["Input", "Reconstruction", "Error"])
        plt.show()

        plt.figure(figsize=(12, 8))
        sns.set(font_scale=2)
        sns.set_style("white")
        sns.histplot(test_loss, bins=50, kde=True, color='red', linewidth=3)
        plt.axvline(x=np.mean(test_loss), color='g', linestyle='--', linewidth=3)
        plt.text(np.mean(test_loss), 30, "Anomaly Mean", horizontalalignment='center',
                 size='small', color='black', weight='semibold')
        plt.text(threshold, 50, "Threshold", horizontalalignment='center',
                 size='small', color='Blue', weight='semibold')
        plt.axvline(x=threshold, color='b', linestyle='--', linewidth=3)
        plt.xlabel("loss")
        plt.ylabel("No of examples")
        sns.despine()
        plt.title("Testing loss the AEVAE for (" + str(epochs) + ") epochs with generation (" + str(
            generation) + ")")
        plt.show()

        plt.figure(figsize=(12, 8))
        sns.set(font_scale=2)
        sns.set_style("white")
        sns.histplot(train_loss, bins=50, kde=True, color='grey', linewidth=3)
        plt.axvline(x=np.mean(train_loss), color='g', linestyle='--', linewidth=3)
        plt.text(np.mean(train_loss), 200, "Normal Mean", horizontalalignment='center',
                 size='small', color='black', weight='semibold')
        plt.axvline(x=threshold, color='b', linestyle='--', linewidth=3)
        plt.text(threshold, 250, "Threshold", horizontalalignment='center',
                 size='small', color='Blue', weight='semibold')

        sns.histplot(test_loss, bins=50, kde=True, color='red', linewidth=3)
        plt.axvline(x=np.mean(test_loss), color='g', linestyle='--', linewidth=3)
        plt.text(np.mean(test_loss), 200, "Anomaly Mean", horizontalalignment='center',
                 size='small', color='black', weight='semibold')
        plt.axvline(x=threshold, color='b', linestyle='--', linewidth=3)
        plt.xlabel("loss")
        plt.ylabel("No of examples")
        sns.despine()
        plt.title("Training loss the AEVAE for (" + str(epochs) + ") epochs with generation (" + str(
            generation) + ")")
        plt.show()

        preds = predict(self.autoencoder, test_data, threshold)
        print_stats(preds, test_labels)

        confusion_matrix = get_clf_eval(test_labels, preds, preds)
        plt.figure(figsize=(8, 6))
        sns.set(font_scale=2)
        sns.set_style("white")
        sns.heatmap(confusion_matrix, cmap='gist_yarg_r', annot=True, fmt='d')
        plt.show()

        # Build the models
        self.encoder.build(input_shape=(None, 140))
        self.decoder.build(input_shape=(None, 8))

        self.encoder.summary()
        self.decoder.summary()
        plot_model(self.encoder, to_file='model_plot.png', show_shapes=True, show_layer_names=True, dpi=96)

    def modifyNetStructure(self):
        self.adaptiveLayerStatus = True
        self.runModifications(self.encoder, self.decoder)

    def runModifications(self, encoderModel, decoderModel):
        new_layer = Dense(32, activation='relu')
        # define the insertion point
        InsertionPoint = 2
        # Insert the layer at the exact insertion point
        modifiedEncoder = encoderModel.layers.insert(InsertionPoint, new_layer)
        modifiedDecoder = decoderModel.layers.insert(InsertionPoint, new_layer)
        self.modelPool.append((modifiedEncoder, modifiedDecoder))
        return modifiedEncoder, modifiedDecoder

        """
        # Check if the model pool is not empty  
        if len(self.modelPool < 0):
            # Save the intial structure of the autoEncoder into the model pool
            self.modelPool.append((self.encoder, self.decoder))
        else:
            # Takes the first last structure of the pool and modify the structure
            lastModel = self.modelPool[-1]

            # Get the encoder/decoder information
            lastModelEncoder = lastModel[0]
            lastModelDecoder = lastModel[1]

            # define the insertion point
            InsertionPoint = 4

            # Insert the layer at the exact insertion point
            modifiedEncoder = lastModelEncoder.insert(InsertionPoint, new_layer)
            modifiedDecoder = lastModelDecoder.insert(InsertionPoint, new_layer)

            # Modify the model pool using the newly generated layers
            """


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

    # Modifies the AE structure
    def modification(self):
        for member in self.population:
            member.modifyNetStructure()

    # The main function for running the evolution
    def run_evolution(self):
        for episode in range(self.generations):
            self.clear_losses()
            self.runEncoderEvolution(episode)
            if episode != self.generations - 1:
                self.normalize()
                # self.reproduction()
                # self.mutate()
                self.modification()


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
