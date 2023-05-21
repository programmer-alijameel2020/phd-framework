import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_curve, roc_curve
from sklearn.metrics import ConfusionMatrixDisplay

from sklearn.model_selection import train_test_split
from keras import layers, losses
from keras.datasets import fashion_mnist
from keras.models import Model

import umap
import umap.plot

from plotly.subplots import make_subplots
import plotly.graph_objects as go


class AnomalyDetector(Model):
    def __init__(self):
        super(AnomalyDetector, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Dense(32, activation="relu"),
            layers.Dense(16, activation="relu"),
            layers.Dense(8, activation="relu")])

        self.decoder = tf.keras.Sequential([
            layers.Dense(16, activation="relu"),
            layers.Dense(32, activation="relu"),
            layers.Dense(140, activation="sigmoid")])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


autoencoder = AnomalyDetector()
autoencoder.compile(optimizer='adam', loss='mae')


def runAutoEncoder(epoches, dataset):

    dataframe = dataset
    raw_data = dataframe.values
    dataframe.head().style.set_properties(**{'background-color': 'black',
                                             'color': 'white',
                                             'border-color': 'white'})

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

    history = autoencoder.fit(normal_train_data, normal_train_data,
                              epochs=epoches,
                              batch_size=512,
                              validation_data=(test_data, test_data),
                              shuffle=True)

    plt.figure(figsize=(10, 8))
    sns.set(font_scale=2)
    sns.set_style("white")
    plt.plot(history.history["loss"], label="Training Loss", linewidth=3.0)
    plt.plot(history.history["val_loss"], label="Validation Loss", linewidth=3.0)
    plt.legend()

    encoded_imgs = autoencoder.encoder(normal_test_data).numpy()
    decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()

    for i in range(0, 3):
        plt.figure(figsize=(10, 8))
        sns.set(font_scale=2)
        sns.set_style("white")
        plt.plot(normal_test_data[i], 'black', linewidth=2)
        plt.plot(decoded_imgs[i], 'red', linewidth=2)
        plt.fill_between(np.arange(140), decoded_imgs[i], normal_test_data[i], color='lightcoral')
        plt.legend(labels=["Input", "Reconstruction", "Error"])
        plt.show()

    encoded_imgs_normal = pd.DataFrame(encoded_imgs)
    encoded_imgs_normal['label'] = 1

    encoded_imgs = autoencoder.encoder(anomalous_test_data).numpy()
    decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()

    for i in range(0, 3):
        plt.figure(figsize=(10, 8))
        sns.set(font_scale=2)
        sns.set_style("white")
        plt.plot(anomalous_test_data[i], 'black', linewidth=2)
        plt.plot(decoded_imgs[i], 'red', linewidth=2)
        plt.fill_between(np.arange(140), decoded_imgs[i], anomalous_test_data[i], color='lightcoral')
        plt.legend(labels=["Input", "Reconstruction", "Error"])
        plt.show()

    encoded_imgs_abnormal = pd.DataFrame(encoded_imgs)
    encoded_imgs_abnormal['label'] = 0

    all_encoded = pd.concat([encoded_imgs_normal, encoded_imgs_abnormal])
    mapper = umap.UMAP().fit(all_encoded.iloc[:, :8])

    reconstructions = autoencoder.predict(normal_train_data)
    train_loss = tf.keras.losses.mae(reconstructions, normal_train_data)
    np.mean(train_loss)

    #umap.plot.points(mapper, labels=all_encoded.iloc[:,8], theme='fire')

    #umap.plot.connectivity(mapper, show_points=True)

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


    reconstructions = autoencoder.predict(anomalous_test_data)
    test_loss = tf.keras.losses.mae(reconstructions, anomalous_test_data)

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
    plt.show()

    preds = predict(autoencoder, test_data, threshold)
    print_stats(preds, test_labels)

    confusion_matrix = get_clf_eval(test_labels, preds, preds)
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=2)
    sns.set_style("white")
    sns.heatmap(confusion_matrix, cmap='gist_yarg_r', annot=True, fmt='d')
    plt.show()

def predict(model, data, threshold):
     reconstructions = model(data)
     loss = tf.keras.losses.mae(reconstructions, data)
     return tf.math.less(loss, threshold)

def print_stats(predictions, labels):
    print("Accuracy = {}".format(accuracy_score(labels, predictions)))
    print("Precision = {}".format(precision_score(labels, predictions)))
    print("Recall = {}".format(recall_score(labels, predictions)))

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











