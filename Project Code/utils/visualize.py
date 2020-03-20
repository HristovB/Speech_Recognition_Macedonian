"""
Utility functions for visualizing data

Copyright 2020 by Blagoj Hristov

See the LICENSE file for the licensing associated with this software.

Author:
  Blagoj Hristov, March 2020

"""

import numpy as np
from matplotlib import pyplot as plt
from utils.utils import get_spectrogram_params


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
