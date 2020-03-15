import os


def increment(number):
    number = int(number) + 1
    return format(number, '04d')


def reset():
    return format(0, '04d')


def rename_files(path, verbose=False):
    """"
    This function is for renaming the files, as per an agreed upon convention, to allow for easier reading.

    Parameters:
        path (string): String variable containing the path to the main data folder (containing multiple folders of batches)
        verbose (bool): Boolean variable to determine whether to print the progress of the function

    Returns:
        None

    """

    folder_list = os.listdir(path)
    file_count = reset()

    if verbose:
        print('Renaming...')
        print()

    for folder in folder_list:
        if verbose:
            print('Loading folder', folder, '...')

        batch_list = sorted(os.listdir(path + os.sep + folder))

        for batch in batch_list:
            if verbose:
                print('Loading batch', batch, '...')

            file_list = sorted(os.listdir(path + os.sep + folder + os.sep + batch))

            for file in file_list:
                if not file.endswith('.wav'):
                    continue

                new_name = folder + '-' + batch + '-' + str(file_count) + '.wav'
                file_count = increment(file_count)

                os.rename(path + os.sep + folder + os.sep + batch + os.sep + file,
                          path + os.sep + folder + os.sep + batch + os.sep + new_name)

            file_count = reset()
            if verbose:
                print('Batch done!')
                print()

        if verbose:
            print('Folder done!')
            print()

    if verbose:
        print('Renaming Successful!')
        print()


if __name__ == '__main__':

    data_path = 'F:\\Speech_Recognition_Macedonian\\Database\\train'

    rename_files(data_path)
