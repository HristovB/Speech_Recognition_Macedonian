# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 23:40:02 2019

@author: Marija
"""
import os
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
import numpy as np
from data_generator import AudioGenerator
from keras import backend as K
from utilities import int_sequence_to_text
from IPython.display import Audio
# import NN architectures for speech recognition
from sample_models import simple_rnn_model, cnn_rnn_model
# import function for training acoustic model
from train_utils import train_model
from keras.layers import Input, GRU, Activation
from keras.models import Model
from tqdm import tqdm
import pandas as pd


os.chdir('d:/DIPLOMSKA') 
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.Session(config=config))
tf.Session(config=config).run(tf.global_variables_initializer())

def simple_rnn_model(input_dim, units, output_dim=34):
    #Build a recurrent network for speech 
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(output_dim, return_sequences=True, 
                 implementation=2, name='rnn')(input_data)
    simp_rnn2 = GRU(units, return_sequences=True, 
                 implementation=2, name='rnn2')(simp_rnn)
    y_pred = Activation('softmax', name='softmax')(simp_rnn2)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def test_models():
    
    units=[32,64,128]
        
    for unit in units:
        
        model_0 = simple_rnn_model(input_dim=161, unit=unit)
        
        train_model(input_to_softmax=model_0, 
                    pickle_path='model_0_'+str(unit)+'.pickle', 
                    save_model_path='model_0'+str(unit)+'.h5',
                    spectrogram=True)
    
test_models()

def get_predictions(index, partition, input_to_softmax, model_path):
    """ Print a model's decoded predictions
    Params:
        index (int): The example you would like to visualize
        partition (str): One of 'train' or 'validation'
        input_to_softmax (Model): The acoustic model
        model_path (str): Path to saved acoustic model's weights
    """
    # load the train and test data
    data_gen = AudioGenerator()
    data_gen.load_train_data()
    data_gen.load_validation_data()
    
    print(len(data_gen.valid_texts))
    # obtain the true transcription and the audio features 
    if partition == 'validation':
        transcr = data_gen.valid_texts[index]
        audio_path = data_gen.valid_audio_paths[index]
        data_point = data_gen.normalize(data_gen.featurize(audio_path))
    elif partition == 'train':
        transcr = data_gen.train_texts[index]
        audio_path = data_gen.train_audio_paths[index]
        data_point = data_gen.normalize(data_gen.featurize(audio_path))
    else:
        raise Exception('Invalid partition!  Must be "train" or "validation"')
        
    # obtain and decode the acoustic model's predictions
    input_to_softmax.load_weights(model_path)
    prediction = input_to_softmax.predict(np.expand_dims(data_point, axis=0))
    output_length = [input_to_softmax.output_length(data_point.shape[0])] 
    pred_ints = (K.eval(K.ctc_decode(
                prediction, output_length)[0][0])+1).flatten().tolist()
    
    if not data_gen.valid_texts[index]:
        return 
    
    with open(r'D:\DIPLOMSKA\results\predictions_cnn_rnn_12.txt', 'a+', encoding='utf8') as fp:
        fp.write('True transcription:\n' + '\n' + transcr+ '\n')
        #print(transcr)
        fp.write('-'*30 + '\n')
        fp.write('Predicted transcription:\n' + '\n' + ''.join(int_sequence_to_text(pred_ints))+ '\n')
        #print(int_sequence_to_text(pred_ints))
        fp.write('-'*30+ '\n')
   
df = pd.read_json(r'D:\DIPLOMSKA\valid_corpus.json',lines=True)

for i in tqdm(range(len(df))):
    get_predictions(index=i, 
                    partition='validation',
                    input_to_softmax=cnn_rnn_model(input_dim=161,
                        filters=8,
                        kernel_size=2, 
                        conv_stride=2,
                        conv_border_mode='valid',
                        units=32), 
                    model_path=r'D:\DIPLOMSKA\results\model_2.h5')