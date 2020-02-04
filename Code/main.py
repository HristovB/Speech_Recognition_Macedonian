from scipy.io import wavfile
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import os
import fnmatch
from python_speech_features import mfcc
from ast import literal_eval
import json


# Uncomment if running on laptop:
import matplotlib
matplotlib.use('GTK3Agg')


def read_wav(path, index):
    wav_data = []
    length = 0
    audio = []
    file_count = 0

    folder_count = len(fnmatch.filter(os.listdir(path), '*'))
    print(folder_count)
    for j in range(1, folder_count + 1):
        if j < 10:
            if index == 1 and j == 4:
                continue

            file_count = len(fnmatch.filter(os.listdir(path + '00000' + str(j)), '*.wav'))

        elif j >= 10:
            if (index == 8 and j == 32) or (index == 9 and j == 39):
                continue

            file_count = len(fnmatch.filter(os.listdir(path + '0000' + str(j)), '*.wav'))

        for i in range(file_count):

            if j < 10:
                if index == 1 and j == 4:
                    continue

                if i < 10:
                    length, audio = wavfile.read(path + '00000' + str(j) + '/' + str(index) + '-00000' + str(j) + '-000' + str(i) + '.wav')

                elif 10 <= i < 100:
                    if index == 9 and i == 39:
                        continue

                    length, audio = wavfile.read(path + '00000' + str(j) + '/' + str(index) + '-00000' + str(j) + '-00' + str(i) + '.wav')

                elif i > 100:
                    length, audio = wavfile.read(path + '00000' + str(j) + '/' + str(index) + '-00000' + str(j) + '-0' + str(i) + '.wav')

            elif j >= 10:
                if index == 8 and j == 32:
                    continue

                if i < 10:
                    length, audio = wavfile.read(path + '0000' + str(j) + '/' + str(index) + '-0000' + str(j) + '-000' + str(i) + '.wav')

                elif 10 <= i < 100:
                    if index == 9 and i == 39:
                        continue

                    length, audio = wavfile.read(path + '0000' + str(j) + '/' + str(index) + '-0000' + str(j) + '-00' + str(i) + '.wav')

                elif i > 100:
                    length, audio = wavfile.read(path + '0000' + str(j) + '/' + str(index) + '-0000' + str(j) + '-0' + str(i) + '.wav')

            wav_data.append([index, length, audio])

    # index = wav_data[0, :]
    wav_data = np.array(wav_data)

    indices = wav_data[:, 0]
    rates = wav_data[:, 1]
    signals = wav_data[:, 2]

    array = signals[0]
    print(array)

    test_mfcc = mfcc(signal=array, samplerate=rates[0], numcep=13)
    # wav_data = pd.DataFrame(wav_data, columns=['Id', 'Signal Rate', 'Signal Data'])

    fig, ax = plt.subplots()
    mfcc_data = np.swapaxes(test_mfcc, 0, 1)
    cax = ax.imshow(mfcc_data, interpolation='nearest', cmap='coolwarm')
    ax.set_title('MFCC')

    plt.show()

    return wav_data


if __name__ == '__main__':

    for idx in range(1, 10):
        if idx == 4 or idx == 5:
            continue

        new_speaker = read_wav(path='/home/bhristov/Documents/Programming/PyCharm/Speech_Recognition_Macedonian/Database/train/' + str(idx) + '/', index=idx)
        # print(new_speaker)
        break


    # for signal in signals:
    #     print(type(signal))
    #     print(int(signal))
    #
    #     # all_data_mfcc = mfcc(signal=signal, samplerate=rate, numcep=13)
    #
    #     # print(all_data_mfcc)