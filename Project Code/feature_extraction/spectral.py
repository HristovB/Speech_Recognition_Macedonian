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
from python_speech_features import mfcc
from utils.utils import find_maximum_all, count_files_batch


def generate_mfcc(path, sampling_rate, num_coeff, verbose=False):
    """
    This function is for generating the mel-frequency cepstral coefficients for each individual batch folder.

    Parameters:
        path (string): String variable containing the path to the main data folder (containing multiple folders, divided into batches of audio files)
        sampling_rate (int): Integer variable containing the sampling rate of the audio(ex: 16kHz ==> sampling_rate = 16000)
        num_coeff (int): Integer variable containing the number of mel-frequency cepstral coefficients to be generated (number of features)
        verbose (bool): Boolean variable to determine whether to print the progress of the function

    Returns:
        Creates a .npy file of the numpy arrays containing the generated MFCCs in the corresponding batch folder;
        Axis 0 represents the generated dada through time;
        Axis 1 represents the individual coefficients;
        Axis 2 represents the multiple individual audio files.

    """

    folder_list = os.listdir(path)

    if verbose:
        print('Generating MFCC...')
        print()

    maximum = find_maximum_all(path=path, sampling_rate=sampling_rate, num_coeff=num_coeff)     # current maximum is 9759 (19.3.2020)

    for folder in folder_list:
        if verbose:
            print('Loading folder', folder, '...')

        batch_list = sorted(os.listdir(path + os.sep + folder))

        for batch in batch_list:
            if verbose:
                print('Loading batch', batch, '...')

            count = count_files_batch(path=path + os.sep + folder + os.sep + batch)

            batch_mfcc = np.empty((maximum, num_coeff, count))

            file_list = sorted(os.listdir(path + os.sep + folder + os.sep + batch))
            num_file = 0

            for file in file_list:
                if not file.endswith('.wav'):
                    continue

                audio, _ = lb.load(path + os.sep + folder + os.sep + batch + os.sep + file, sr=sampling_rate)

                mfcc_data = mfcc(signal=audio, samplerate=sampling_rate, numcep=num_coeff)

                mean = np.mean(mfcc_data)
                std = np.std(mfcc_data)
                mfcc_data = (mfcc_data - mean)/std

                mfcc_data = np.pad(mfcc_data, ((0, maximum - len(mfcc_data)), (0, 0)), 'constant', constant_values=0.)

                batch_mfcc[:, :, num_file] = mfcc_data

                num_file = num_file + 1

            np.save(path + os.sep + folder + os.sep + batch + os.sep + folder + '-' + batch + '-mfcc', batch_mfcc)

            if verbose:
                print('Batch', batch, 'done!')
                print()

        if verbose:
            print('Folder', folder, 'done!')
            print()

    if verbose:
        print('Generation Successful!')
        print()
