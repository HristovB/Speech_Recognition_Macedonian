import os


def increment(number):
    number = int(number) + 1
    return format(number, '04d')


def reset():
    return format(0, '04d')


if __name__ == '__main__':

    path = 'F:\\Speech_Recognition_Macedonian\\Database\\train'
    folder_list = os.listdir(path)
    file_count = reset()

    for folder in folder_list:
        batch_list = sorted(os.listdir(path + os.sep + folder))

        for batch in batch_list:
            file_list = sorted(os.listdir(path + os.sep + folder + os.sep + batch))

            for file in file_list:
                if not file.endswith('.wav'):
                    continue

                new_name = folder + '-' + batch + '-' + str(file_count) + '.wav'
                file_count = increment(file_count)
                os.rename(path + os.sep + folder + os.sep + batch + os.sep + file,
                          path + os.sep + folder + os.sep + batch + os.sep + new_name)

            file_count = reset()
            print('Next batch')

        print('Next folder')
