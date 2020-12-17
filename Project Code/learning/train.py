"""
Functions used during training and pre-training of Keras models.

Copyright 2020 by Blagoj Hristov

See the LICENSE file for the licensing associated with this software.

Author:
  Blagoj Hristov, March 2020

"""

import tensorflow as tf


def ctc_loss(logits, labels, logit_length, label_length):
    """
    This function is for calculating the value of the connectionist temporal classification (CTC) loss function.

    Parameters:
        logits: Logits from the output dense layer
        labels: Labels converted to array of indices
        logit_length: Array containing length of each input in the batch
        label_length: Array containing length of each label in the batch

    Returns:
        Array of ctc loss for each element in batch

    """

    return tf.nn.ctc_loss(labels=labels, logits=logits, label_length=label_length, logit_length=logit_length, logits_time_major=False, unique=None, blank_index=-1, name=None)


def train_file(x, y, optimizer, model):
    """
    This function is for training the model on a single sample (audio file)

    Parameters:
        x : NumPy array containing the training data (spectrogram/MFCC)
        y : List variable containing the enumerated transcript file for the audio sample
        model (Keras model): Generated Keras model
        optimizer (Keras optimizer): Optimizer to be used during training

    Returns:
        None

    """

    with tf.GradientTape() as tape:
        logits = model(x)
        labels = y
        logits_length = [logits.shape[1]]*logits.shape[0]
        labels_length = [labels.shape[1]]*labels.shape[0]
        loss = ctc_loss(logits, labels, logit_length=logits_length, label_length=labels_length)
        loss = tf.reduce_mean(loss)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return loss


def train_batch(model, optimizer, X, Y, epochs):

    for step in range(1, epochs):
        loss = train_file(X, Y, optimizer, model)
        print('Epoch {}, Loss: {}'.format(step, loss))