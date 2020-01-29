# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 18:26:42 2019

@author: Marija
"""
import create_desc_json
import os
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf 
from models import *
from train_utils import train_model

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
set_session(tf.Session(config=config))
tf.Session(config=config).run(tf.global_variables_initializer())

os.chdir('F:\Speech_Recognition_Macedonian')

# Create json files

# create_desc_json.main(os.getcwd() + os.sep + 'Database' + os.sep + 'train', 'train_corpus.json')
# create_desc_json.main(os.getcwd() + os.sep + 'Database' + os.sep + 'test', 'valid_corpus.json')

# The type of recurrent unit (LSTM or GRU) is changed in models.py

# model = cnn_rnn_model(input_dim=161, filters=30, kernel_size=2, conv_stride=1, conv_border_mode='valid', units=50)

model = simple_rnn_model(input_dim=(None, 161))

train_model(input_to_softmax=model,
            pickle_path='model_cnn_rnn_12_50gru_30f.pickle',
            save_model_path='model_cnn_rnn_12_50gru_30f.h5',
            spectrogram=True)  # change to False if you would like to use MFCC features
