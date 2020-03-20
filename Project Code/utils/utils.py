"""
General utility functions

Copyright 2020 by Blagoj Hristov

See the LICENSE file for the licensing associated with this software.

Author:
  Blagoj Hristov, March 2020

"""

import numpy as np
import re


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


def get_char_set():
    """
    This function is for generating and returning the array of possible characters to be identified by the speech recognition algorithm (all letters of the Macedonian alphabet).

    Returns:
        alphabet (list): List variable containing the letters of the Macedonian alphabet

    """

    return ['а', 'б', 'в', 'г', 'д', 'ѓ', 'е', 'ж', 'з', 'ѕ', 'и', 'ј', 'к', 'л', 'љ', 'м', 'н', 'њ', 'о', 'п', 'р', 'с', 'т', 'ќ', 'у', 'ф', 'х', 'ц', 'ч', 'џ', 'ш']
