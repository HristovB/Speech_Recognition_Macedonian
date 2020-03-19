"""
Functions for refactoring of audio transcripts as per agreed upon naming convention

Copyright 2020 by Blagoj Hristov

See the LICENSE file for the licensing associated with this software.

Author:
  Blagoj Hristov, March 2020

"""

import os
import regex.regex as re
from utils.utils import increment_file, reset_file, is_indexed


def rename_transcripts(path, verbose=False):
    """
    This function is for renaming the transcript text files, as per an agreed upon convention, to allow for easier reading.

    Parameters:
        path (string): String variable containing the path to the main data folder (containing multiple folders of literature works, which contain multiple folders of batches of audio)
        verbose (bool): Boolean variable to determine whether to print the progress of the function

    Returns:
        None

    """

    folder_list = os.listdir(path)

    if verbose:
        print('Indexing...')
        print()

    for folder in folder_list:
        if verbose:
            print('Loading folder', folder, '...')

        batch_list = sorted(os.listdir(path + os.sep + folder))

        for batch in batch_list:
            if verbose:
                print('Loading batch', batch, '...')

            file_list = sorted([file for file in os.listdir(path + os.sep + folder + os.sep + batch) if file.endswith('.txt')])

            if not file_list:
                continue

            if len(file_list) > 1:
                for extra_file in file_list[1:]:
                    os.remove(path + os.sep + folder + os.sep + batch + os.sep + extra_file)
                file_list = [file_list[0]]

            new_name = folder + '-' + batch + '-' + 'trans' + '.txt'

            os.rename(path + os.sep + folder + os.sep + batch + os.sep + file_list[0],
                      path + os.sep + folder + os.sep + batch + os.sep + new_name)

            if verbose:
                print('Batch', batch, 'done!')
                print()

        if verbose:
            print('Folder', folder, 'done!')
            print()

    if verbose:
        print('Renaming Successful!')
        print()


def index_transcript(path, folder, batch):
    """
    This function is for proper indexing a single transcript file, as per an agreed upon convention, to allow for easier reading.

    Parameters:
        path (string): String variable containing the path to the transcript .txt file
        folder (string): String variable of the name of the folder containing multiple batch folders (used for naming convention)
        batch (string): String variable of the name of the batch folder containing the transcript .txt file to be indexed (used for naming convention)

    Returns:
        Creates a new .txt file containing the indexed contents of the original

    """

    file_count = reset_file()

    src = open(path, mode='r', encoding='utf-8')
    dst = open(path[:-4] + '-indexed.txt', mode='w', encoding='utf-8')

    transcript = src.read()

    transcript_array = [element.split(' ', 1) for element in transcript.strip().split('\n')]

    if is_indexed(transcript_array):
        transcript_array = [element.split(' ', 1) for element in transcript.strip().split('\n')]
    else:
        transcript_array = [[element] for element in transcript.strip().split('\n')]

    num_lines = len(transcript_array) - 1
    count = 0

    for row in transcript_array:
        transcript = row[-1]
        indexing = folder + '-' + batch + '-' + file_count

        if count == num_lines:
            new_indexing = indexing + ' ' + transcript
        else:
            new_indexing = indexing + ' ' + transcript + '\n'

        dst.write(new_indexing)

        file_count = increment_file(file_count)
        count = count + 1

    src.close()
    dst.close()


def format_transcript(path):
    """
    This function is for converting the transcript text to all uppercase letters and removing all special characters (except for whitespace and newline characters), as per agreed upon convention.

    Parameters:
        path (string): String variable containing the path to the transcript .txt file

    Returns:
        Creates a new .txt file containing the formatted contents of the original

    """

    src = open(path, mode='r', encoding='utf-8')
    dst = open(path[:-4] + '-formatted.txt', mode='w', encoding='utf-8')

    transcript = src.read()

    transcript_array = [element.split(' ', 1) for element in transcript.strip().split('\n')]

    if is_indexed(transcript_array):
        transcript = re.sub('[^\\p{L} \n\\d-]', '', transcript)
    else:
        transcript = re.sub('[^\\p{L} \n]', '', transcript)

    transcript = transcript.upper()

    dst.write(transcript)

    src.close()
    dst.close()


def refactor_all(path, verbose=False):
    """
    This function is for proper formatting and indexing of all of the transcript text files located in the main data folder, as per an agreed upon convention, to allow for easier reading.

    Parameters:
        path (string): String variable containing the path to the main data folder (containing multiple folders of literature works, which contain multiple folders of batches of audio)
        verbose (bool): Boolean variable to determine whether to print the progress of the function

    Returns:
        None

    """

    folder_list = os.listdir(path)

    if verbose:
        print('Indexing...')
        print()

    for folder in folder_list:
        if verbose:
            print('Loading folder', folder, '...')

        batch_list = sorted(os.listdir(path + os.sep + folder))

        for batch in batch_list:
            if verbose:
                print('Loading batch', batch, '...')

            file_list = sorted([file for file in os.listdir(path + os.sep + folder + os.sep + batch) if file.endswith('.txt')])

            if not file_list:
                continue

            format_transcript(path=path + os.sep + folder + os.sep + batch + os.sep + file_list[0])

            file_list = sorted([file for file in os.listdir(path + os.sep + folder + os.sep + batch) if file.endswith('.txt')])

            index_transcript(path=path + os.sep + folder + os.sep + batch + os.sep + file_list[0], folder=folder, batch=batch)

            if verbose:
                print('Batch', batch, 'done!')
                print()

        if verbose:
            print('Folder', folder, 'done!')
            print()

    rename_transcripts(path)

    if verbose:
        print('Refactoring Successful!')
        print()
