import os
from utils.utils import increment_file, reset_file


def rename_transcripts(path, verbose=False):
    """"
    This function is for renaming the transcript text files, as per an agreed upon convention, to allow for easier reading.

    Parameters:
        path (string): String variable containing the path to the main data folder (containing multiple folders of batches)
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

            if len(file_list) == 2:
                original_name = file_list[0]
                os.remove(path + os.sep + folder + os.sep + batch + os.sep + file_list[0])
                os.rename(path + os.sep + folder + os.sep + batch + os.sep + file_list[1], path + os.sep + folder + os.sep + batch + os.sep + original_name)

            new_name = folder + '-' + batch + '-' + 'trans' + '.txt'

            os.rename(path + os.sep + folder + os.sep + batch + os.sep + file_list[0],
                      path + os.sep + folder + os.sep + batch + os.sep + new_name)

            if verbose:
                print('Batch done!')
                print()

        if verbose:
            print('Folder done!')
            print()

    if verbose:
        print('Renaming Successful!')
        print()


def index_transcripts(path, verbose=False):
    """"
    This function is for proper indexing of the transcript text files, as per an agreed upon convention, to allow for easier reading.

    Parameters:
        path (string): String variable containing the path to the main data folder (containing multiple folders of batches)
        verbose (bool): Boolean variable to determine whether to print the progress of the function

    Returns:
        None

    """

    folder_list = os.listdir(path)
    file_count = reset_file()

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

            src = open(path + os.sep + folder + os.sep + batch + os.sep + file_list[0], mode='r', encoding='utf-8')

            if len(file_list) == 1:
                dst = open(path + os.sep + folder + os.sep + batch + os.sep + 'new_' + file_list[0], mode='w', encoding='utf-8')

            else:
                dst = open(path + os.sep + folder + os.sep + batch + os.sep + file_list[1], mode='w', encoding='utf-8')

            file_contents = src.read()

            transcript_array = [element.split(' ', 1) for element in file_contents.strip().split('\n')]

            num_files = len(transcript_array) - 1
            count = 0

            for row in transcript_array:
                transcript = row[-1]
                indexing = folder + '-' + batch + '-' + file_count

                if count == num_files:
                    new_indexing = indexing + ' ' + transcript
                else:
                    new_indexing = indexing + ' ' + transcript + '\n'

                dst.write(new_indexing)

                file_count = increment_file(file_count)
                count = count + 1

            src.close()
            dst.close()
            file_count = reset_file()

            if verbose:
                print('Batch done!')
                print()

        if verbose:
            print('Folder done!')
            print()

    rename_transcripts(path)

    if verbose:
        print('Indexing Successful!')
        print()
