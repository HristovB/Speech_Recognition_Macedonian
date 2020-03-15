import os
import librosa as lb


def resample_audio(path, sampling_rate, verbose=False):
    """"
    This function is for resampling audio files to a set sampling rate.

    Parameters:
        path (string): String variable containing the path to a main folder (containing multiple batches of audio)
        sampling_rate (int): Integer variable containing the value of the desired sampling rate (ex: 16kHz ==> sampling_rate = 16000)
        verbose (bool): Boolean variable to determine whether to print the progress of the function

    Returns:
        None

    """

    if verbose:
        print('Resampling...')
        print()

    batch_list = sorted(os.listdir(path))

    for batch in batch_list:
        if verbose:
            print('Loading next batch...')

        file_list = sorted(os.listdir(path + os.sep + batch))

        for file in file_list:
            if not file.endswith('.wav'):
                continue

            audio, _ = lb.load(path + os.sep + batch + os.sep + file, sr=sampling_rate)

            lb.output.write_wav(path + os.sep + batch + os.sep + file, audio, sr=sampling_rate)

        if verbose:
            print('Batch done!')
            print()

    if verbose:
        print('Resampling Successful!')
        print()


if __name__ == '__main__':

    data_path = 'F:\\Speech_Recognition_Macedonian\\Database\\train'
    rate = 16000

    for folder in os.listdir(data_path):
        print('Loading folder', folder, '...')

        resample_audio(path=data_path + os.sep + folder, sampling_rate=rate)

        print('Folder', folder, 'done!')
        print()
