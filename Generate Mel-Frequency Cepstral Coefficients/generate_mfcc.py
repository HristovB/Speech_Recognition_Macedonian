import os
import librosa as lb
import numpy as np
from python_speech_features import mfcc


def find_maximum(path, sampling_rate, num_coeff, verbose=False):
    """"
    This function is for finding the maximum possible length of the generated MFCC features (longest audio clip), for determining the size of the data matrix.

    With the initial data, the maximum of the whole dataset is: 9759

    Parameters:
        path (string): String variable containing the path to the main data folder (containing multiple folders of batches)
        sampling_rate (int): Integer value to determine the desired sampling rate (ex: 16kHz ==> sampling_rate = 16000)
        num_coeff (int): Number of mel-frequency cepstral coefficients to be generated (number of features)
        verbose (bool): Boolean variable to determine whether to print the progress of the function

    Returns:
        max_length (int): The maximum possible length of the audio files

    """

    folder_list = os.listdir(path)
    max_length = 0

    if verbose:
        print('Finding maximum...')
        print()

    for directory in folder_list:
        if verbose:
            print('Loading folder', directory, '...')

        batch_list = sorted(os.listdir(path + os.sep + directory))

        for batch in batch_list:
            if verbose:
                print('Loading batch', batch, '...')

            file_list = sorted(os.listdir(path + os.sep + directory + os.sep + batch))

            for file in file_list:
                if not file.endswith('.wav'):
                    continue

                audio, _ = lb.load(path + os.sep + directory + os.sep + batch + os.sep + file, sr=sampling_rate)

                max_length = max(max_length, len(mfcc(signal=audio, samplerate=sampling_rate, numcep=num_coeff)))

            if verbose:
                print('Batch done!')

        if verbose:
            print('Folder done!')
            print()

    if verbose:
        print('Maximum:', max_length)
        print()

    return max_length


def find_batch_maximum(path, sampling_rate, num_coeff, verbose=False):
    """"
    This function is for finding the maximum possible length of the generated MFCC features (longest audio clip), for determining the size of the data matrix.

    With the initial data, the maximum of the whole dataset is: 9759

    Parameters:
        path (string): String variable containing the path to the main data folder (containing multiple folders of batches)
        sampling_rate (int): Integer value to determine the desired sampling rate (ex: 16kHz ==> sampling_rate = 16000)
        num_coeff (int): Number of mel-frequency cepstral coefficients to be generated (number of features)
        verbose (bool): Boolean variable to determine whether to print the progress of the function

    Returns:
        max_length (int): The maximum possible length of the audio files

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
        print('Maximum:', max_length)

    return max_length


def file_count(path, verbose=False):
    """"
    This function is for counting the total number of audio files.

    With the initial data, the count of the whole dataset is: 12401

    Parameters:
        path (string): String variable containing the path to the main data folder (containing multiple folders of batches)
        verbose (bool): Boolean variable to determine whether to print the progress of the function

    Returns:
        count (int): The total number of audio files in the dataset

    """

    folder_list = os.listdir(path)
    count = 0

    if verbose:
        print('Counting...')
        print()

    for directory in folder_list:
        if verbose:
            print('Loading folder', directory, '...')

        batch_list = sorted(os.listdir(path + os.sep + directory))

        for batch in batch_list:
            if verbose:
                print('Loading batch', batch, '...')

            file_list = sorted(os.listdir(path + os.sep + directory + os.sep + batch))

            for file in file_list:
                if not file.endswith('.wav'):
                    continue

                count = count + 1

            if verbose:
                print('Batch done!')

        if verbose:
            print('Folder done!')
            print()

    if verbose:
        print('Count:', count)
        print()

    return count


def generate_mfcc(path, sampling_rate, num_coeff, folder_name, verbose=False):
    """"
    This function is for generating the mel-frequency cepstral coefficients.

    Parameters:
        path (string): String variable containing the path to the main data folder (containing multiple folders of batches)
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
        maximum = find_batch_maximum(path=path + os.sep + batch, sampling_rate=rate, num_coeff=13, verbose=True)

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


if __name__ == '__main__':

    data_path = 'F:\\Speech_Recognition_Macedonian\\Database\\train'
    rate = 16000
    coeff = 13

    # mfcc_dat = np.load(data_path + os.sep + '1' + os.sep + '000001' + os.sep + '1-000001_mfcc.npy')

    for folder in os.listdir(data_path):
        print('Loading folder', folder, '...')

        generate_mfcc(path=data_path + os.sep + folder, sampling_rate=rate, num_coeff=coeff, folder_name=folder, verbose=True)

        print('Folder', folder, 'done!')
        print()
