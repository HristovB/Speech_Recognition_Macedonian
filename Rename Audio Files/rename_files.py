import os


def increment(number):
    number = int(number) + 1
    return format(number, '04d')


def reset():
    return format(0, '04d')


def rename_files(path):
    """"
    This function is for renaming the files, as per an agreed upon convention, to allow for easier reading.

    Parameters:
        path (string): String variable containing the path to the main data folder (containing multiple folders of batches)

    Returns:
        None

    """

    folder_list = os.listdir(path)
    file_count = reset()

    print('Renaming...')

    for folder in folder_list:
        print('Loading folder', folder, '...')

        batch_list = sorted(os.listdir(path + os.sep + folder))

        for batch in batch_list:
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
            print('Batch done!')

        print('Folder done!')
        print()

    print('Renaming Successful!')


if __name__ == '__main__':

    data_path = 'F:\\Speech_Recognition_Macedonian\\Database\\train'

    rename_files(data_path)
