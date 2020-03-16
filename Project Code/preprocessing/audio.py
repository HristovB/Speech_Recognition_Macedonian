import os
import librosa as lb


def resample_audio(path, sampling_rate, verbose=False):
    """"
    This function is for resampling audio files to a set sampling rate.

    Parameters:
        path (string): String variable containing the path to the main data folder (containing multiple folders of literature works, which contain multiple folders of batches of audio)
        sampling_rate (int): Integer variable containing the value of the desired sampling rate (ex: 16kHz ==> sampling_rate = 16000)
        verbose (bool): Boolean variable to determine whether to print the progress of the function

    Returns:
        None

    """

    folder_list = os.listdir(path)

    if verbose:
        print('Resampling...')
        print()

    for folder in folder_list:
        if verbose:
            print('Loading folder', folder, '...')

        batch_list = sorted(os.listdir(path + os.sep + folder))

        for batch in batch_list:
            if verbose:
                print('Loading next batch...')

            file_list = sorted(os.listdir(path + os.sep + folder + os.sep + batch))

            for file in file_list:
                if not file.endswith('.wav'):
                    continue

                audio, _ = lb.load(path + os.sep + folder + os.sep + batch + os.sep + file, sr=sampling_rate)

                lb.output.write_wav(path + os.sep + folder + os.sep + batch + os.sep + file, audio, sr=sampling_rate)

            if verbose:
                print('Batch', batch, 'done!')
                print()

        if verbose:
            print('Folder', folder, 'done!')
            print()

    if verbose:
        print('Resampling Successful!')
        print()
