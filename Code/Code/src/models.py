from keras import backend as K
from keras.models import Model
from keras.layers import (BatchNormalization, Conv1D, Dense, Input, TimeDistributed, Activation, SimpleRNN, GRU, LSTM)

# The models without CNN do not work for spectrogram input.
# Maybe they should be tried with MFCC features.


def simple_rnn_model(input_dim, output_dim=34):
    """ Build a simple recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=input_dim)

    # Add recurrent layer
    simp_rnn = LSTM(output_dim, return_sequences=True, implementation=2, name='rnn')(input_data)

    y_pred = Activation(activation='softmax', name='softmax')(simp_rnn)

    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x

    return model


def rnn_model(input_dim, units, activation, output_dim=34):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Recurrent layer
    simp_rnn = LSTM(units, activation=activation,
        return_sequences=True, implementation=2, name='rnn')(input_data)
    # Batch normalization 
    bn_rnn = BatchNormalization(name='bn_rnn_1d')(simp_rnn)
    # Time distributed dense
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


def cnn_rnn_model(input_dim, filters, kernel_size, conv_stride, conv_border_mode, units, output_dim=34):

    """ Build a recurrent + convolutional network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Convolutional layer

    conv_1d = Conv1D(filters, kernel_size,
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)

    # Batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
    # Recurrent layer
    simp_rnn = LSTM(units, activation='relu',
        return_sequences=True, implementation=2, name='rnn')(bn_cnn)
    # Normalization
    bn_rnn = BatchNormalization(name='bn_rnn_1d')(simp_rnn)
    # Time distributed dense
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Softmax
    y_pred = Activation('softmax', name='softmax')(time_dense)
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(x, kernel_size, conv_border_mode, conv_stride)
    #print(model.summary())
    return model


def cnn_output_length(input_length, filter_size, border_mode, stride, dilation=1):
    """ Compute the length of the output sequence after 1D convolution along
        time. Note that this function is in line with the function used in
        Convolution1D class from Keras.
    Params:
        input_length (int): Length of the input sequence.
        filter_size (int): Width of the convolution kernel.
        border_mode (str): Only support `same` or `valid`.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
    """
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - dilated_filter_size + 1
    return (output_length + stride - 1) // stride


def deep_rnn_model(input_dim, units, recur_layers, output_dim=34):
    """ Build a deep recurrent network for speech 
    """
    input_data = Input(name='the_input', shape=(None, input_dim))
    if recur_layers == 1:
        layer = LSTM(units, return_sequences=True, activation='relu')(input_data)
        layer = BatchNormalization(name='bt_rnn_1')(layer)
    else:
        layer = LSTM(units, return_sequences=True, activation='relu')(input_data)
        layer = BatchNormalization(name='bt_rnn_1')(layer)

        for i in range(recur_layers - 2):
            layer = LSTM(units, return_sequences=True, activation='relu')(layer)
            layer = BatchNormalization(name='bt_rnn_{}'.format(2+i))(layer)

        layer = LSTM(units, return_sequences=True, activation='relu')(layer)
        layer = BatchNormalization(name='bt_rnn_last_rnn')(layer)

    time_dense = TimeDistributed(Dense(output_dim))(layer)
    y_pred = Activation('softmax', name='softmax')(time_dense)
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model