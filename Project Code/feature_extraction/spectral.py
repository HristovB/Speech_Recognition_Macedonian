"""
Functions for spectrogram based feature extraction

Copyright 2020 by Blagoj Hristov

See the LICENSE file for the licensing associated with this software.

Author:
  Blagoj Hristov, March 2020

"""

import os
import librosa as lb
import numpy as np
import h5py
from scipy import signal
from python_speech_features import mfcc
from utils.utils import find_maximum_all, count_files_batch
from preprocessing.spectral import normalize, padding


def generate_spectrogram(path, sampling_rate, verbose=False):
    """
    This function is for generating the frequency spectrogram for each audio file in each individual batch folder.

    Parameters:
        path (string): String variable containing the path to the main data folder (containing multiple folders, divided into batches of audio files)
        sampling_rate (int): Integer variable containing the value of the audio sampling rate (ex: 16kHz ==> sampling_rate = 16000)
        verbose (bool): Boolean variable to determine whether to print the progress of the function

    Returns:
        Creates a .h5 file of the NumPy arrays containing the generated spectrogram in the corresponding batch folder
        Axis 0 represents the data through time
        Axis 1 represents the frequency
        Axis 2 represents the multiple individual audio files

    """

    folder_list = os.listdir(path)

    if verbose:
        print('Generating Spectrogram...')
        print()

    maximum = find_maximum_all(path=path, sampling_rate=sampling_rate, method='spectrogram', verbose=True)      # current maximum is 6971 (19.03.2020)

    for folder in folder_list:
        if verbose:
            print('Loading folder', folder, '...')

        batch_list = sorted(os.listdir(path + os.sep + folder))

        for batch in batch_list:
            if verbose:
                print('Loading batch', batch, '...')

            count = count_files_batch(path=path + os.sep + folder + os.sep + batch)

            h5_file = h5py.File(name=path + os.sep + folder + os.sep + batch + os.sep + folder + '-' + batch + '-spectrogram.h5', mode='w', libver='latest')

            batch_spectrogram = h5_file.create_dataset(name='Spectrogram', shape=(maximum, 129, count), chunks=(maximum, 129, 1), dtype=np.float32, compression='lzf')

            file_list = sorted(os.listdir(path + os.sep + folder + os.sep + batch))
            num_file = 0

            for file in file_list:
                if not file.endswith('.wav'):
                    continue

                audio, _ = lb.load(path + os.sep + folder + os.sep + batch + os.sep + file, sr=sampling_rate)

                _, _, spectrogram_data = signal.spectrogram(x=audio, fs=sampling_rate)

                with np.errstate(divide='ignore'):
                    spectrogram_data = np.swapaxes(10*np.log10(spectrogram_data), 0, 1)
                    spectrogram_data[np.isneginf(spectrogram_data)] = 0.0

                spectrogram_data = normalize(spectrogram_data)
                spectrogram_data = padding(spectrogram_data, maximum)

                batch_spectrogram[:, :, num_file] = spectrogram_data

                num_file = num_file + 1

            h5_file.close()

            if verbose:
                print('Batch', batch, 'done!')
                print()

        if verbose:
            print('Folder', folder, 'done!')
            print()

    if verbose:
        print('Generation Successful!')
        print()


def generate_mfcc(path, sampling_rate, num_coeff, verbose=False):
    """
    This function is for generating the mel-frequency cepstral coefficients for each audio file for each individual batch folder.

    Parameters:
        path (string): String variable containing the path to the main data folder (containing multiple folders, divided into batches of audio files)
        sampling_rate (int): Integer variable containing the sampling rate of the audio(ex: 16kHz ==> sampling_rate = 16000)
        num_coeff (int): Integer variable containing the number of mel-frequency cepstral coefficients to be generated (number of features)
        verbose (bool): Boolean variable to determine whether to print the progress of the function

    Returns:
        Creates a .h5 file of the numpy arrays containing the generated MFCCs in the corresponding batch folder
        Axis 0 represents the generated data through time
        Axis 1 represents the mel-frequency cepstral coefficients
        Axis 2 represents the multiple individual audio files

    """

    folder_list = os.listdir(path)

    if verbose:
        print('Generating MFCC...')
        print()

    # maximum = find_maximum_all(path=path, sampling_rate=sampling_rate,  method='mfcc', num_coeff=num_coeff)     # current maximum is 9759 (19.03.2020)
    maximum = 9759

    for folder in folder_list:
        if verbose:
            print('Loading folder', folder, '...')

        batch_list = sorted(os.listdir(path + os.sep + folder))

        for batch in batch_list:
            if verbose:
                print('Loading batch', batch, '...')

            count = count_files_batch(path=path + os.sep + folder + os.sep + batch)

            h5_file = h5py.File(name=path + os.sep + folder + os.sep + batch + os.sep + folder + '-' + batch + '-mfcc.h5', mode='w', libver='latest')

            batch_mfcc = h5_file.create_dataset(name='MFCC', shape=(maximum, num_coeff, count), chunks=(maximum, num_coeff, 1), dtype=np.float32, compression='lzf')

            file_list = sorted(os.listdir(path + os.sep + folder + os.sep + batch))
            num_file = 0

            for file in file_list:
                if not file.endswith('.wav'):
                    continue

                audio, _ = lb.load(path + os.sep + folder + os.sep + batch + os.sep + file, sr=sampling_rate)

                mfcc_data = mfcc(signal=audio, samplerate=sampling_rate, numcep=num_coeff)

                mfcc_data = normalize(mfcc_data)
                mfcc_data = padding(mfcc_data, maximum)

                batch_mfcc[:, :, num_file] = mfcc_data

                num_file = num_file + 1

            h5_file.close()

            if verbose:
                print('Batch', batch, 'done!')
                print()

        if verbose:
            print('Folder', folder, 'done!')
            print()

    if verbose:
        print('Generation Successful!')
        print()
