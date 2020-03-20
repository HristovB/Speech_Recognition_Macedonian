"""
Utility functions for working with data

Copyright 2020 by Blagoj Hristov

See the LICENSE file for the licensing associated with this software.

Author:
  Blagoj Hristov, March 2020

"""

import os
import h5py
import re
import numpy as np
from scipy import signal
import librosa as lb
from python_speech_features import mfcc


def find_maximum_batch(path, sampling_rate, method='spectrogram', num_coeff=None, verbose=False):
    """
    This function is for finding the length of the longest audio clip (through the spectrogram or MFCC features) in a batch folder, for determining the size of the data array.

    Parameters:
        path (string): String variable containing the path to a batch folder (containing multiple audio files)
        sampling_rate (int): Integer variable containing the value of the audio sampling rate (ex: 16kHz ==> sampling_rate = 16000)
        method (string): {'spectrogram', 'mfcc'} String variable to determine whether to search maximum for spectrogram or MFCC features
        num_coeff (int): Number of mel-frequency cepstral coefficients to be generated (number of features) - only when using 'mfcc' method!
        verbose (bool): Boolean variable to determine whether to print the progress of the function

    Returns:
        max_length (int): The length of the longest audio file in the batch

    """

    file_list = sorted(os.listdir(path))
    max_length = 0

    if verbose:
        print('Finding maximum...')

    for file in file_list:
        if not file.endswith('.wav'):
            continue

        audio, _ = lb.load(path + os.sep + file, sr=sampling_rate)

        if method == 'mfcc':
            max_length = max(max_length, len(mfcc(signal=audio, samplerate=sampling_rate, numcep=num_coeff)))

        elif method == 'spectrogram':
            _, _, spectrogram_data = signal.spectrogram(x=audio, fs=sampling_rate)
            spectrogram_data = np.swapaxes(10 * np.log10(spectrogram_data), 0, 1)

            max_length = max(max_length, len(spectrogram_data))

        else:
            raise ValueError('Wrong input for method argument! Possible inputs: \'spectrogram\', \'mfcc\'')

    if verbose:
        print('Batch maximum:', max_length)

    return max_length


def find_maximum_folder(path, sampling_rate, method='spectrogram', num_coeff=None, verbose=False):
    """
    This function is for finding the length of the longest audio clip (through the spectrogram or MFCC features) in a main folder, for determining the size of the data array.

    Parameters:
        path (string): String variable containing the path to a main folder (containing multiple batch folders)
        sampling_rate (int): Integer variable containing the value of the audio sampling rate (ex: 16kHz ==> sampling_rate = 16000)
        method (string): {'spectrogram', 'mfcc'} String variable to determine whether to search maximum for spectrogram or MFCC features
        num_coeff (int): Number of mel-frequency cepstral coefficients to be generated (number of features) - only when using 'mfcc' method!
        verbose (bool): Boolean variable to determine whether to print the progress of the function

    Returns:
        max_length (int): The length of the longest audio file in the folder

    """

    batch_list = sorted(os.listdir(path))
    max_length = 0

    if verbose:
        print('Finding maximum...')
        print()

    for batch in batch_list:
        if verbose:
            print('Loading batch', batch, '...')

        max_length = max(max_length, find_maximum_batch(path=path + os.sep + batch, sampling_rate=sampling_rate, method=method, num_coeff=num_coeff))

        if verbose:
            print('Batch', batch, 'done!')
            print()

    if verbose:
        print('Folder maximum:', max_length)
        print()

    return max_length


def find_maximum_all(path, sampling_rate, method='spectrogram', num_coeff=None, verbose=False):
    """
    This function is for finding the length of the longest audio clip (through the spectrogram or MFCC features) the entire dataset, for determining the size of the data array.

    Parameters:
        path (string): String variable containing the path to a main folder (containing multiple batch folders)
        sampling_rate (int): Integer variable containing the value of the audio sampling rate (ex: 16kHz ==> sampling_rate = 16000)
        method (string): {'spectrogram', 'mfcc'} String variable to determine whether to search maximum for spectrogram or MFCC features
        num_coeff (int): Number of mel-frequency cepstral coefficients to be generated (number of features) - only when using 'mfcc' method!
        verbose (bool): Boolean variable to determine whether to print the progress of the function

    Returns:
        max_length (int): The length of the longest audio file in the entire dataset

    """

    folder_list = os.listdir(path)
    max_length = 0

    if verbose:
        print('Finding maximum...')
        print()

    for folder in folder_list:
        if verbose:
            print('Loading folder', folder, '...')

        max_length = max(max_length, find_maximum_folder(path=path + os.sep + folder, sampling_rate=sampling_rate, method=method, num_coeff=num_coeff))

        if verbose:
            print('Folder', folder, 'done!')
            print()

    if verbose:
        print('Maximum:', max_length)
        print()

    return max_length


def count_files_folder(path, verbose=False):
    """
    This function is for counting the total number of audio files in a folder.

    Parameters:
        path (string): String variable containing the path to a main folder (containing multiple batch folders)
        verbose (bool): Boolean variable to determine whether to print the progress of the function

    Returns:
        count (int): The total number of audio files in the dataset

    """

    batch_list = sorted(os.listdir(path))
    count = 0

    if verbose:
        print('Counting...')
        print()

    for batch in batch_list:
        if verbose:
            print('Loading batch', batch, '...')

        file_list = sorted(os.listdir(path + os.sep + batch))

        for file in file_list:
            if not file.endswith('.wav'):
                continue

            count = count + 1

        if verbose:
            print('Batch', batch, 'done!')
            print()

    if verbose:
        print('Total count:', count)
        print()

    return count


def count_files_batch(path, verbose=False):
    """
    This function is for counting the total number of audio files in a batch.

    Parameters:
        path (string): String variable containing the path to a batch folder (containing multiple audio files)
        verbose (bool): Boolean variable to determine whether to print the progress of the function

    Returns:
        count (int): The total number of audio files in the batch

    """

    file_list = sorted(os.listdir(path))
    count = 0

    if verbose:
        print('Counting...')
        print()

    for file in file_list:
        if not file.endswith('.wav'):
            continue

        count = count + 1

    if verbose:
        print('Count:', count)
        print()

    return count


def load_mfcc_batch(path):
    """
    This function is for batchwise loading of the generated MFCC features into a 3D NumPy array.

    Parameters:
        path (string): String variable containing the path to a batch folder (containing multiple audio files)

    Returns:
        batch_mfcc_data (np.ndarray): 3D NumPy array containing the 2D spectrogram features for all audio files in the batch folder
        Axis 0 represents the data through time
        Axis 1 represents the mel-frequency cepstral coefficients
        Axis 2 represents the multiple individual audio files

    """

    data_files = [file for file in os.listdir(path) if file.endswith('.h5')]
    mfcc_file = None

    for file in data_files:
        if bool(re.match(r'﻿?[0-9]-[0-9]{6}-mfcc\.h5', file)):
            mfcc_file = file

    hdf5_file = h5py.File(name=path + os.sep + mfcc_file, mode='r')
    batch_mfcc_data = hdf5_file['MFCC'][:]

    return batch_mfcc_data


def load_spectrogram_batch(path):
    """
    This function is for batchwise loading of the generated spectrogram features into a 3D NumPy array.

    Parameters:
        path (string): String variable containing the path to a batch folder (containing multiple audio files)

    Returns:
        batch_spectrogram_data (np.ndarray): 3D NumPy array containing the 2D spectrogram features for all audio files in the batch folder
        Axis 0 represents the data through time
        Axis 1 represents the frequency
        Axis 2 represents the multiple individual audio files

    """

    data_files = [file for file in os.listdir(path) if file.endswith('.h5')]
    spectrogram_file = None

    for file in data_files:
        if bool(re.match(r'﻿?[0-9]-[0-9]{6}-spectrogram\.h5', file)):
            spectrogram_file = file

    hdf5_file = h5py.File(name=path + os.sep + spectrogram_file, mode='r')
    batch_spectrogram_data = hdf5_file['Spectrogram'][:]

    return batch_spectrogram_data


def load_transcript(path):
    """
    This function is for batchwise loading of the transcripts of the audio files in the batch folder.

    Parameters:
        path (string): String variable containing the path to a batch folder (containing multiple audio files)

    Returns:
        batch_transcripts (list): List variable containing the transcripts (string) of the audio files in the batch folder

    """

    file_name= [file for file in os.listdir(path) if file.endswith('.txt')][0]

    transcript_file = open(path + os.sep + file_name, mode='r', encoding='utf-8')
    batch_transcripts = [line.split(' ', 1)[1] for line in transcript_file.read().split('\n')]
    transcript_file.close()

    return batch_transcripts
