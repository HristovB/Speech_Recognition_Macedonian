import os
import librosa as lb
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


def count_files(path, verbose=False):
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


def increment_batch(number):
    """"
    This function is for proper incrementing of the batch folder names, as per agreed upon convention.

    Parameters:
        number (int): Integer variable containing the name (number) of the current batch folder

    Returns:
        number: The number incremented by 1 in the proper format, as per agreed upon convention

    """

    number = int(number) + 1
    return format(number, '06d')


def reset_batch():
    """"
    This function is for resetting the counter for the batch folder names, as per agreed upon convention.

    Returns:
        number: The reset counter, equal to zero, in proper format as per agreed upon convention.

    """

    return format(0, '06d')


def increment_file(number):
    """"
    This function is for proper incrementing of the file names, as per agreed upon convention.

    Parameters:
        number (int): Integer variable containing the name (number) of the current file

    Returns:
        number: The number incremented by 1 in the proper format, as per agreed upon convention

    """

    number = int(number) + 1
    return format(number, '04d')


def reset_file():
    """"
    This function is for resetting the counter for the file names, as per agreed upon convention.

    Returns:
        number: The reset counter, equal to zero, in proper format as per agreed upon convention.

    """

    return format(0, '04d')
