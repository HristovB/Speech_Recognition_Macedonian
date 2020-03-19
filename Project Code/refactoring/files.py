"""
Functions for refactoring of audio files and folders as per agreed upon naming convention

Copyright 2020 by Blagoj Hristov

See the LICENSE file for the licensing associated with this software.

Author:
  Blagoj Hristov, March 2020

"""

import os
from natsort import natsorted, ns
from utils.utils import increment_file, reset_file, increment_batch, reset_batch


def rename_files(path, verbose=False):
    """
    This function is for renaming the files, as per an agreed upon convention, to allow for easier reading.

    Parameters:
        path (string): String variable containing the path to the main data folder (containing multiple folders of literature works, which contain multiple folders of batches of audio)
        verbose (bool): Boolean variable to determine whether to print the progress of the function

    Returns:
        None

    """

    folder_count = 1
    batch_count = reset_batch()
    file_count = reset_file()

    folder_list = os.listdir(path)

    if verbose:
        print('Renaming...')
        print()

    for folder in folder_list:
        if verbose:
            print('Loading folder', folder, '...')

        new_folder_name = str(folder_count)
        folder_count = folder_count + 1

        os.rename(path + os.sep + folder, path + os.sep + new_folder_name)

        batch_list = sorted(os.listdir(path + os.sep + new_folder_name))

        for batch in batch_list:
            if verbose:
                print('Loading batch', batch, '...')

            new_batch_name = str(batch_count)
            batch_count = increment_batch(batch_count)

            os.rename(path + os.sep + new_folder_name + os.sep + batch, path + os.sep + new_folder_name + os.sep + new_batch_name)

            file_list = os.listdir(path + os.sep + new_folder_name + os.sep + new_batch_name)
            file_list = natsorted(file_list, alg=ns.PATH | ns.IGNORECASE)

            for file in file_list:
                if not file.endswith('.wav'):
                    if file.endswith('mfcc.h5'):
                        new_name = new_folder_name + '-' + new_batch_name + '-mfcc' + '.h5'
                        os.rename(path + os.sep + new_folder_name + os.sep + new_batch_name + os.sep + file,
                                  path + os.sep + new_folder_name + os.sep + new_batch_name + os.sep + new_name)

                    if file.endswith('spectrogram.h5'):
                        new_name = new_folder_name + '-' + new_batch_name + '-spectrogram' + '.h5'
                        os.rename(path + os.sep + new_folder_name + os.sep + new_batch_name + os.sep + file,
                                  path + os.sep + new_folder_name + os.sep + new_batch_name + os.sep + new_name)

                    if file.endswith('.mp3'):
                        new_name = new_folder_name + '-' + new_batch_name + '.mp3'
                        os.rename(path + os.sep + new_folder_name + os.sep + new_batch_name + os.sep + file,
                                  path + os.sep + new_folder_name + os.sep + new_batch_name + os.sep + new_name)
                    continue

                new_name = new_folder_name + '-' + new_batch_name + '-' + str(file_count) + '.wav'
                file_count = increment_file(file_count)

                os.rename(path + os.sep + new_folder_name + os.sep + new_batch_name + os.sep + file,
                          path + os.sep + new_folder_name + os.sep + new_batch_name + os.sep + new_name)

            file_count = reset_file()
            if verbose:
                print('Batch', batch, 'done!')
                print()

        batch_count = reset_batch()
        if verbose:
            print('Folder', folder, 'done!')
            print()

    if verbose:
        print('Renaming Successful!')
        print()
