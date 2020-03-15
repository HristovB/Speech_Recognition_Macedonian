import os
import librosa as lb


def resample_audio(path):

    print('Resampling...')

    batch_list = sorted(os.listdir(path))

    for batch in batch_list:
        print('Loading next batch...')
        file_list = sorted(os.listdir(path + os.sep + batch))

        for file in file_list:
            if not file.endswith('.wav'):
                continue

            audio, _ = lb.load(path + os.sep + batch + os.sep + file, sr=16000)

            lb.output.write_wav(path + os.sep + batch + os.sep + file, audio, sr=16000)

        print('Batch done!')

    print('Resampling Successful!')


if __name__ == '__main__':

    resample_audio(path='F:\\Speech_Recognition_Macedonian\\Database\\train\\2')
