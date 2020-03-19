"""
Utility functions

Copyright 2020 by Blagoj Hristov

See the LICENSE file for the licensing associated with this software.

Author:
  Blagoj Hristov, March 2020

"""

import os
import numpy as np
import librosa as lb
import re
from matplotlib import pyplot as plt
from python_speech_features import mfcc


def find_maximum_batch(path, sampling_rate, num_coeff, verbose=False):
    """
    This function is for finding the length of the longest audio clip (through the MFCC features) in a batch folder, for determining the size of the data array.

    Parameters:
        path (string): String variable containing the path to a batch folder (containing multiple audio files)
        sampling_rate (int): Integer value to determine the desired sampling rate (ex: 16kHz ==> sampling_rate = 16000)
        num_coeff (int): Number of mel-frequency cepstral coefficients to be generated (number of features)
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

        max_length = max(max_length, len(mfcc(signal=audio, samplerate=sampling_rate, numcep=num_coeff)))

    if verbose:
        print('Batch maximum:', max_length)

    return max_length


def find_maximum_folder(path, sampling_rate, num_coeff, verbose=False):
    """
    This function is for finding the length of the longest audio clip (through the MFCC features) in a main folder, for determining the size of the data array.

    Parameters:
        path (string): String variable containing the path to a main folder (containing multiple batch folders)
        sampling_rate (int): Integer value to determine the desired sampling rate (ex: 16kHz ==> sampling_rate = 16000)
        num_coeff (int): Number of mel-frequency cepstral coefficients to be generated (number of features)
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

        max_length = max(max_length, find_maximum_batch(path + os.sep + batch, sampling_rate=sampling_rate, num_coeff=num_coeff))

        if verbose:
            print('Batch', batch, 'done!')
            print()

    if verbose:
        print('Folder maximum:', max_length)
        print()

    return max_length


def find_maximum_all(path, sampling_rate, num_coeff, verbose=False):
    """
    This function is for finding the length of the longest audio clip (through the MFCC features) the entire dataset, for determining the size of the data array.

    Parameters:
        path (string): String variable containing the path to a main folder (containing multiple batch folders)
        sampling_rate (int): Integer value to determine the desired sampling rate (ex: 16kHz ==> sampling_rate = 16000)
        num_coeff (int): Number of mel-frequency cepstral coefficients to be generated (number of features)
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

        max_length = max(max_length, find_maximum_folder(path + os.sep + folder, sampling_rate=sampling_rate, num_coeff=num_coeff))

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


def increment_batch(number):
    """
    This function is for proper incrementing of the batch folder names, as per agreed upon convention.

    Parameters:
        number (int): Integer variable containing the name (number) of the current batch folder

    Returns:
        number: The number incremented by 1 in the proper format, as per agreed upon convention

    """

    number = int(number) + 1
    return format(number, '06d')


def reset_batch():
    """
    This function is for resetting the counter for the batch folder names, as per agreed upon convention.

    Returns:
        number: The reset counter, equal to zero, in proper format as per agreed upon convention.

    """

    return format(0, '06d')


def increment_file(number):
    """
    This function is for proper incrementing of the file names, as per agreed upon convention.

    Parameters:
        number (int): Integer variable containing the name (number) of the current file

    Returns:
        number: The number incremented by 1 in the proper format, as per agreed upon convention

    """

    number = int(number) + 1
    return format(number, '04d')


def reset_file():
    """
    This function is for resetting the counter for the file names, as per agreed upon convention.

    Returns:
        number: The reset counter, equal to zero, in proper format as per agreed upon convention.

    """

    return format(0, '04d')


def is_indexed(transcript):
    """
    This function is for checking if the transcript files already contain indexing (old dataset), as per agreed upon convention.

    Parameters:
        transcript (list): List variable containing the transcript file text, divided  by new-line characters

    Returns:
        bool: Boolean value as to whether the indexing is present or not

    """

    if bool(re.match(r'﻿?[0-9]-[0-9]{6}-[0-9]{4}', transcript[0][0])):
        return True
    else:
        return False


def load_mfcc_batch(path):
    """
    This function is for batchwise loading of the generated MFCC features into a list of 2D NumPy matrices.

    Parameters:
        path (string): String variable containing the path to a batch folder (containing multiple audio files)

    Returns:
        batch_mfcc (list): List variable containing the 2D MFCC features for all audio files in the batch folder

    """

    file = [file for file in os.listdir(path) if file.endswith('.npy')]

    mfcc_data = np.load(path + os.sep + file[0])

    return mfcc_data


def plot_mfcc_batch(mfcc_data):
    """
    This function is for plotting the generated MFCC features of a single audio file.

    Parameters:
        mfcc_data (string): 2D numpy array containing the generated MFCC features (axis 0 ==> data through time; axis 1 ==> coefficients)

    Returns:
        Plots the MFCC matrix in a new window

    """

    mfcc_data = mfcc_data[~np.all(mfcc_data == 0, axis=1)]
    mfcc_data = np.swapaxes(mfcc_data, 0, 1)

    plt.figure(figsize=(18, 4))
    plt.imshow(mfcc_data, interpolation='nearest', aspect='auto')
    plt.title('Mel-frequency cepstral coefficients')
    plt.xlabel('Time (ms)')
    plt.ylabel('Coefficients')
    plt.show()
