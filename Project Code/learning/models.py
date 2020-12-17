"""
Functions for defining Keras models to be used for speech recognition.

Copyright 2020 by Blagoj Hristov

See the LICENSE file for the licensing associated with this software.

Author:
  Blagoj Hristov, March 2020

"""

import tensorflow as tf


def baseline_bilstm(input_shape, lstm_units, output_size=34):
    """
    This function is for generating a simple Bidirectional LSTM Neural Network model for baseline results.

    Parameters:
        input_shape (tuple): Tuple variable containing the shape of the data that will be sent as an input to the network
        lstm_units (int): Integer variable to determine the size of the LSTM layers
        output_size (int): Integer variable containing the expected size of the output of the network (default is 34, as there are 31 letters in the Macedonian alphabet + the whitespace character, blank token and end token)

    Returns:
        model (keras model): Keras model of the generated BiLSTM Neural Network to be used for training

    """

    input_layer = tf.keras.layers.Input(name='input_layer', shape=input_shape)

    conv_layer_1 = tf.keras.layers.Conv1D(name='conv1D_1', kernel_size=8, strides=2, padding='valid', filters=256, activation='relu')(input_layer)
    norm_1 = tf.keras.layers.BatchNormalization()(conv_layer_1)

    conv_layer_2 = tf.keras.layers.Conv1D(name='conv1D_2', kernel_size=8, strides=2, padding='valid', filters=256, activation='relu')(norm_1)
    norm_2 = tf.keras.layers.BatchNormalization()(conv_layer_2)

    conv_layer_3 = tf.keras.layers.Conv1D(name='conv1D_3', kernel_size=8, strides=2, padding='valid', filters=256, activation='relu')(norm_2)
    norm_3 = tf.keras.layers.BatchNormalization()(conv_layer_3)

    lstm_forward_1 = tf.keras.layers.GRU(name='lstm_f1', units=lstm_units, return_sequences=True, activation='tanh')
    lstm_backward_1 = tf.keras.layers.GRU(name='lstm_b1', units=lstm_units, return_sequences=True, activation='tanh', go_backwards=True)

    bilstm_layer_1 = tf.keras.layers.Bidirectional(name='bilstm_1', layer=lstm_forward_1, backward_layer=lstm_backward_1)(norm_3)
    norm_4 = tf.keras.layers.BatchNormalization()(bilstm_layer_1)

    lstm_forward_2 = tf.keras.layers.GRU(name='lstm_f2', units=lstm_units, return_sequences=True, activation='tanh')
    lstm_backward_2 = tf.keras.layers.GRU(name='lstm_b2', units=lstm_units, return_sequences=True, activation='tanh', go_backwards=True)

    bilstm_layer_2 = tf.keras.layers.Bidirectional(name='bilstm_2', layer=lstm_forward_2, backward_layer=lstm_backward_2)(norm_4)
    norm_5 = tf.keras.layers.BatchNormalization()(bilstm_layer_2)

    lstm_forward_3 = tf.keras.layers.GRU(name='lstm_f3', units=lstm_units, return_sequences=True, activation='tanh')
    lstm_backward_3 = tf.keras.layers.GRU(name='lstm_b3', units=lstm_units, return_sequences=True, activation='tanh', go_backwards=True)

    bilstm_layer_3 = tf.keras.layers.Bidirectional(name='bilstm_3', layer=lstm_forward_3, backward_layer=lstm_backward_3)(norm_5)
    norm_6 = tf.keras.layers.BatchNormalization()(bilstm_layer_3)

    lstm_forward_4 = tf.keras.layers.GRU(name='lstm_f4', units=lstm_units, return_sequences=True, activation='tanh')
    lstm_backward_4 = tf.keras.layers.GRU(name='lstm_b4', units=lstm_units, return_sequences=True, activation='tanh', go_backwards=True)

    bilstm_layer_4 = tf.keras.layers.Bidirectional(name='bilstm_4', layer=lstm_forward_4, backward_layer=lstm_backward_4)(norm_6)
    norm_7 = tf.keras.layers.BatchNormalization()(bilstm_layer_4)

    output_layer = tf.keras.layers.Dense(name='output_layer', units=output_size, activation='softmax')(norm_7)

    model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

    return model
