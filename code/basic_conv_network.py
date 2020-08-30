import keras
from keras.models import Sequential
from keras.layers import Conv2D, Conv2DTranspose
from keras.constraints import max_norm
from tensorflow.keras.optimizers import Adam

import configurations

# basic model
no_epochs = 250 #500
validation_split = 0.2
max_norm_value = 2.0
batch_size = 50 #20

# Max norm constraints
# Another form of regularization is to enforce an absolute upper bound on the magnitude of the weight
# vector for every neuron and use projected gradient descent to enforce the constraint.

# validation_split
# Float between 0 and 1. Fraction of the training data to be used as validation data.
# The model will set apart this fraction of the training data, will not train on it, and will evaluate the loss and any model metrics on this data at the end of each epoch.
# The validation data is selected from the last samples in the x and y data provided, before shuffling.

class Basic:

    def __init__(self, train_config):

        self.train_config = train_config

    def compile(self):
        model = Sequential()
        conv_size = (5, 5)
        input_shape = (self.train_config.img_width, self.train_config.img_height, self.train_config.channels)
        model.add(Conv2D(64, kernel_size=conv_size, kernel_constraint=max_norm(max_norm_value), activation='relu',
                         kernel_initializer='he_uniform', input_shape=input_shape))
        model.add(Conv2D(32, kernel_size=conv_size, kernel_constraint=max_norm(max_norm_value), activation='relu',
                         kernel_initializer='he_uniform'))
        model.add(Conv2D(16, kernel_size=conv_size, kernel_constraint=max_norm(max_norm_value), activation='relu',
                         kernel_initializer='he_uniform'))
        model.add(Conv2D(4, kernel_size=conv_size, kernel_constraint=max_norm(max_norm_value), activation='relu',
                         kernel_initializer='he_uniform'))
        model.add(
            Conv2DTranspose(4, kernel_size=conv_size, kernel_constraint=max_norm(max_norm_value), activation='relu',
                            kernel_initializer='he_uniform'))
        model.add(
            Conv2DTranspose(16, kernel_size=conv_size, kernel_constraint=max_norm(max_norm_value), activation='relu',
                            kernel_initializer='he_uniform'))
        model.add(
            Conv2DTranspose(32, kernel_size=conv_size, kernel_constraint=max_norm(max_norm_value), activation='relu',
                            kernel_initializer='he_uniform'))
        model.add(
            Conv2DTranspose(64, kernel_size=conv_size, kernel_constraint=max_norm(max_norm_value), activation='relu',
                            kernel_initializer='he_uniform'))
        model.add(
            Conv2D(self.train_config.channels, kernel_size=conv_size, kernel_constraint=max_norm(max_norm_value),
                   activation='sigmoid',
                   padding='same'))

        model.summary()

        # compile model
        model.compile(loss='mae', optimizer='adam', metrics=['mse', 'mae', 'mape'])

        # 1.optimizer = tf.keras.optimizers.RMSprop(0.001)
        # 1.model.compile(optimizer=optimizer, loss=keras_custom_loss_function)
        # model.compile(optimizer='adam', loss='binary_crossentropy')
        return model

    def train(self, model, noisy_input_train, pure_input_train):
        model.fit(noisy_input_train, pure_input_train,
                  epochs=no_epochs,
                  batch_size=batch_size,
                  validation_data=(noisy_input_train, pure_input_train),
                  validation_split=validation_split)