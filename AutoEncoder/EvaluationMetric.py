from keras import backend as K
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, precision_score, recall_score, roc_auc_score
import keras


class evaluationMetric:
    def f1_m(self, y_true, y_pred):
        return f1_score(y_true, y_pred)

    def confusion(self, y_true, y_pred):
        return confusion_matrix(y_true, y_pred)

    def accuracy(self, y_true, y_pred):
        return accuracy_score(y_true, y_pred)

    def precision(self, y_true, y_pred):
        return precision_score(y_true, y_pred)

    def recall(self, y_true, y_pred):
        return recall_score(y_true, y_pred)

    def roc_auc(self, y_true, y_pred):
        return roc_auc_score(y_true, y_pred)

    def meanSquaredError(self, y_true, y_pred):
        mse = keras.losses.MeanSquaredError()
        return mse(y_true, y_pred)
