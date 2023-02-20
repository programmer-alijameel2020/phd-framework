from keras import Sequential
from keras.layers import Conv1D, MaxPooling1D, BatchNormalization, Dense, Flatten, Convolution2D, MaxPooling2D, Dropout


class Network:
    def __init__(self):
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
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def return_acc_history(self):
        return self.acc_history

    def get_layer_weight(self, i):
        return self.model.layers[i].get_weights()

    def set_layer_weight(self, i, weight):
        self.model.layers[i].set_weights(weight)

    def train(self):
        his = self.model.fit(X_train, y_train, epochs=15, batch_size=32,
                             validation_data=(X_test, y_test), callbacks=[logger])
        return his

    def test(self):
        loss, acc = self.model.evaluate(X_test, y_test)
        self.acc_history.append(acc)
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

def CNN_Model():
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

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
