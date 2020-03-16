import os
import librosa as lb
import numpy as np
from python_speech_features import mfcc
from utils import find_batch_maximum


def generate_mfcc(path, sampling_rate, num_coeff, folder_name, verbose=False):
    """"
    This function is for generating the mel-frequency cepstral coefficients.

    Parameters:
        path (string): String variable containing the path to the main data folder (containing multiple folders of batches of audio)
        sampling_rate (int): Integer variable containing the sampling rate of the audio(ex: 16kHz ==> sampling_rate = 16000)
        num_coeff (int): Integer variable containing the number of mel-frequency cepstral coefficients to be generated (number of features)
        folder_name (string): String variable containing the name of the folder where the multiple batch folders are located in
        verbose (bool): Boolean variable to determine whether to print the progress of the function

    Returns:
        None, but saves the numpy arrays containing the generated MFCCs as .npy files in the corresponding batch folder

    """

    if verbose:
        print('Generating MFCC...')
        print()

    batch_list = sorted(os.listdir(path))

    for batch in batch_list:
        if verbose:
            print('Loading next batch...')

        file_list = sorted(os.listdir(path + os.sep + batch))

        count = len(file_list) - 1
        maximum = find_batch_maximum(path=path + os.sep + batch, sampling_rate=sampling_rate, num_coeff=13, verbose=True)

        batch_mfcc = np.empty((maximum, 13, count))
        num_file = 0

        for file in file_list:
            if not file.endswith('.wav'):
                continue

            audio, _ = lb.load(path + os.sep + batch + os.sep + file, sr=sampling_rate)

            mfcc_data = mfcc(signal=audio, samplerate=sampling_rate, numcep=num_coeff)
            mfcc_data = np.pad(mfcc_data, ((0, maximum - len(mfcc_data)), (0, 0)), 'constant', constant_values=np.nan)

            batch_mfcc[:, :, num_file] = mfcc_data

            num_file = num_file + 1

        np.save(path + os.sep + batch + os.sep + folder_name + '-' + batch + '_mfcc', batch_mfcc)

        if verbose:
            print('Batch done!')
            print()

    if verbose:
        print('Generation Successful!')
        print()
