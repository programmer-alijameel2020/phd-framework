from keras import backend as K
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, precision_score, recall_score, roc_auc_score
import keras

class evaluationMetric:

    def recall_m(self, y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision_m(self, y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    def f1_measure(self, y_true, y_pred):
        precision = self.precision_m(y_true, y_pred)
        recall = self.recall_m(y_true, y_pred)
        return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

    def meanSquaredError(self, y_true, y_pred):
        mse = keras.losses.MeanSquaredError()
        return mse(y_true, y_pred)


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
        y_true_np = K.eval(y_true)
        y_pred_np = K.eval(y_pred)
        return roc_auc_score(y_true_np, y_pred_np)

    def meanSquaredError(self, y_true, y_pred):
        mse = keras.losses.MeanSquaredError()
        return mse(y_true, y_pred)
