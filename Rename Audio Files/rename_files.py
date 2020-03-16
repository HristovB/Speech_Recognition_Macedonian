import os


def increment_batch(number):
    number = int(number) + 1
    return format(number, '06d')


def reset_batch():
    return format(0, '06d')


def increment_file(number):
    number = int(number) + 1
    return format(number, '04d')


def reset_file():
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

            file_list = sorted(os.listdir(path + os.sep + new_folder_name + os.sep + new_batch_name))

            for file in file_list:
                if not file.endswith('.wav'):
                    continue

                new_name = new_folder_name + '-' + new_batch_name + '-' + str(file_count) + '.wav'
                file_count = increment_file(file_count)

                os.rename(path + os.sep + new_folder_name + os.sep + new_batch_name + os.sep + file,
                          path + os.sep + new_folder_name + os.sep + new_batch_name + os.sep + new_name)

            file_count = reset_file()
            if verbose:
                print('Batch done!')
                print()

        batch_count = reset_batch()
        if verbose:
            print('Folder done!')
            print()

    if verbose:
        print('Renaming Successful!')
        print()


if __name__ == '__main__':

    data_path = 'F:\\Speech_Recognition_Macedonian\\Database\\train'

    rename_files(data_path)
