# Importing Libraries

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.callbacks import EarlyStopping
from NeuralNetwork.AutoEncoder import Autoencoder


def run_autoEncoder(dataset, epochs, batch_size, stopping_patience=2):
    df = pd.read_csv(dataset, sep='  ', header=None, engine='python')
    # df = pd.read_csv('storage/dataset/test.txt', sep='  ', header=None, engine='python')
    print(df.head())
    df.columns
    print(df.describe())
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

    print("Anomaly training data: ", anomaly_train_data)

    # Making pandas dataframe for the normal and anomaly test data points
    normal_test_data = pd.DataFrame(test_data_scaled).add_prefix('c').query('c0 == 0').values[:, 1:]
    anomaly_test_data = pd.DataFrame(test_data_scaled).add_prefix('c').query('c0 > 0').values[:, 1:]

    # Instantiating the Autoencoder
    model = Autoencoder()

    # creating an early_stopping
    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=stopping_patience,
                                   mode='min')

    # Compiling the model
    model.compile(optimizer='adam',
                  loss='mae')

    # Training the model
    history = model.fit(normal_train_data, normal_train_data,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(train_data_scaled[:, 1:], train_data_scaled[:, 1:]),
                        shuffle=True,
                        callbacks=[early_stopping])

    # predictions for normal test data points
    encoder_out = model.encoder(normal_test_data).numpy()
    decoder_out = model.decoder(encoder_out).numpy()

    print("Test data:", encoder_out)
    print("Predictions:", decoder_out)
    print(decoder_out.shape)
    print(normal_test_data[0], 'b')
    print(decoder_out[0], 'r')

    # predictions for anomaly test data points
    encoder_out_a = model.encoder(anomaly_test_data).numpy()
    decoder_out_a = model.decoder(encoder_out_a).numpy()

    print("Anomaly training data", anomaly_train_data[0], 'b')
    print("Decoder out: ", decoder_out_a[0], 'r')

    # Plotting the normal and anomaly losses with the threshold
    plt.plot(encoder_out_a[0], label="encoder out")
    plt.plot(decoder_out_a[0], label="decoder out")
    plt.title("Abnormality detection report for (" + str(epochs) + ") epochs")
    plt.show()


    # reconstruction loss for normal test data
    reconstructions = model.predict(normal_test_data)
    train_loss = tf.keras.losses.mae(reconstructions, normal_test_data)

    # Plotting histogram for recontruction loss for normal test data
    print("Training loss", train_loss)
    print("Mean: ", np.mean(train_loss))
    print("Standard deviation: ", np.std(train_loss))

    # reconstruction loss for anomaly test data
    reconstructions_a = model.predict(anomaly_test_data)
    train_loss_a = tf.keras.losses.mae(reconstructions_a, anomaly_test_data)

    # Plotting histogram for reconstruction loss for anomaly test data
    print("Reconstructed training loss: ", train_loss_a)
    print("Mean: ", np.mean(train_loss_a))
    print("Standard deviation: ", np.std(train_loss_a))

    # setting threshold
    threshold = np.mean(train_loss) + 2 * np.std(train_loss)
    print("Threshold: ", threshold)

    print("normal_test_data: ", normal_test_data)
    print("Predictions: ", reconstructions)


    # Plotting the normal and anomaly losses with the threshold
    plt.hist(train_loss, bins=50, density=True, label="Normal (train data loss)", alpha=.6, color="green")
    plt.hist(train_loss_a, bins=50, density=True, label="Anomaly (test data loss)", alpha=.6, color="red")
    plt.axvline(threshold, color='r', linewidth=3, linestyle='dashed', label='Threshold value: {:0.3f}'.format(threshold))
    plt.legend(loc='upper right')
    plt.title("Abnormality detection report for (" + str(epochs) + ") epochs")
    plt.show()

    # Number of correct predictions for Normal test data
    preds = tf.math.less(train_loss, threshold)

    print("Number of correct predictions: ", tf.math.count_nonzero(preds))
    # Number of correct predictions for Anomaly test data
    preds_a = tf.math.greater(train_loss_a, threshold)
    print("Number of correct predictions for anomaly data: ", tf.math.count_nonzero(preds_a))
    print(preds_a.shape)
