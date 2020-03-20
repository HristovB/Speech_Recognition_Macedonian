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
import h5py
from scipy import signal
from matplotlib import pyplot as plt
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
        number: The reset counter, equal to zero, in proper format as per agreed upon convention

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
        number: The reset counter, equal to zero, in proper format as per agreed upon convention

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


def plot_mfcc(mfcc_data):
    """
    This function is for plotting the generated MFCC features of a single audio file.

    Parameters:
        mfcc_data (np.ndarray): 2D NumPy array containing the generated MFCC features (axis 0 ==> data through time; axis 1 ==> mel-frequency cepstral coefficients)

    Returns:
        Plots the MFCC 2D array in a new window

    """

    mfcc_data = mfcc_data[~np.all(mfcc_data == 0, axis=1)]
    mfcc_data = np.swapaxes(mfcc_data, 0, 1)

    plt.figure(figsize=(18, 4))
    plt.pcolormesh(mfcc_data)
    plt.title('Mel-frequency cepstral coefficients')
    plt.xlabel('Time [ms]')
    plt.ylabel('Coefficients')
    plt.show()


def get_spectrogram_params(audio_signal, spectrogram_data, sampling_rate):
    """
    This function is for generating the frequency and time parameters of the spectrogram used during plotting.

    Parameters:
        audio_signal (np.ndarray): 2D NumPy array containing the raw audio signal
        spectrogram_data (np.ndarray): 2D NumPy array containing the generated spectrogram (axis 0 ==> data through time; axis 1 ==> frequency)
        sampling_rate (int): Integer variable containing the value of the audio sampling rate (ex: 16kHz ==> sampling_rate = 16000)

    Returns:
        freq (np.ndarray): NumPy array containing the sample frequencies
        time (np.ndarray): NumPy array containing the segment times

    """

    audio_length = len(audio_signal)

    freq_start = 0
    freq_step = sampling_rate / 2 / 128
    freq_stop = sampling_rate / 2 + freq_step
    freq = np.arange(freq_start, freq_stop, freq_step, dtype=np.float64)

    time_start = 1 / sampling_rate * 128
    time_step = round(int(audio_length / spectrogram_data.shape[1]) / sampling_rate, 3)
    time_stop = audio_length / sampling_rate
    time = np.arange(time_start, time_stop, time_step, dtype=np.float64)

    return freq, time


def plot_spectrogram(audio_signal, spectrogram_data, sampling_rate):
    """
    This function is for plotting the generated spectrogram of a single audio file.

    Parameters:
        audio_signal (np.ndarray): 2D NumPy array containing the raw audio signal
        spectrogram_data (np.ndarray): 2D NumPy array containing the generated spectrogram (axis 0 ==> data through time; axis 1 ==> frequency)
        sampling_rate (int): Integer variable containing the value of the audio sampling rate (ex: 16kHz ==> sampling_rate = 16000)

    Returns:
        Plots the spectrogram 2D array in a new window

    """

    spectrogram_data = spectrogram_data[~np.all(spectrogram_data == 0, axis=1)]
    spectrogram_data = np.swapaxes(spectrogram_data, 0, 1)

    freq, time = get_spectrogram_params(audio_signal=audio_signal, spectrogram_data=spectrogram_data, sampling_rate=sampling_rate)

    plt.figure(figsize=(18, 4))
    plt.pcolormesh(time, freq, spectrogram_data)
    plt.title('Spectrogram')
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [Hz]')
    plt.show()


def plot_all(audio_signal, spectrogram_data, mfcc_data, sampling_rate):
    """
    This function is for plotting the audio signal, generated spectrogram and generated MFCC features on the same figure, for comparison.

    Parameters:
        audio_signal (np.ndarray): 2D NumPy array containing the raw audio signal
        spectrogram_data (np.ndarray): 2D NumPy array containing the generated spectrogram (axis 0 ==> data through time; axis 1 ==> frequency)
        mfcc_data (np.ndarray): 2D NumPy array containing the generated MFCC features (axis 0 ==> data through time; axis 1 ==> mel-frequency cepstral coefficients)
        sampling_rate (int): Integer variable containing the value of the audio sampling rate (ex: 16kHz ==> sampling_rate = 16000)

    Returns:
        Plots the audio signal, spectrogram and MFCC features as subplots on the same figure in a new window

    """

    spectrogram_data = spectrogram_data[~np.all(spectrogram_data == 0, axis=1)]
    spectrogram_data = np.swapaxes(spectrogram_data, 0, 1)

    mfcc_data = mfcc_data[~np.all(mfcc_data == 0, axis=1)]
    mfcc_data = np.swapaxes(mfcc_data, 0, 1)

    fig, ax = plt.subplots(nrows=3, ncols=1)
    fig.tight_layout()

    print(len(audio_signal))
    ax[0].plot(audio_signal)
    ax[0].set_title('Audio Signal')
    ax[0].set_xlabel('Sample number')
    ax[0].set_ylabel('Amplitude')

    freq, time = get_spectrogram_params(audio_signal=audio_signal, spectrogram_data=spectrogram_data, sampling_rate=sampling_rate)

    ax[1].pcolormesh(time, freq, spectrogram_data)
    ax[1].set_title('Spectrogram')
    ax[1].set_xlabel('Time [s]')
    ax[1].set_ylabel('Frequency [Hz]')

    ax[2].pcolormesh(mfcc_data)
    ax[2].set_title('Mel-frequency cepstral coefficients')
    ax[2].set_xlabel('Time [ms]')
    ax[2].set_ylabel('Coefficients')

    plt.show()
