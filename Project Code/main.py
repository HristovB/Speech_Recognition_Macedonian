import os
import numpy as np
from feature_extraction.spectral import generate_mfcc
from refactoring.transcript import index_transcripts
from refactoring.files import rename_files
from preprocessing.audio import resample_audio


if __name__ == '__main__':

    data_path = 'F:\\Speech_Recognition_Macedonian\\Database\\train'
    rate = 16000
    coeff = 13

    # mfcc_dat = np.load(data_path + os.sep + '1' + os.sep + '000001' + os.sep + '1-000001_mfcc.npy')

    # rename_files(path=data_path, verbose=True)

    generate_mfcc(path=data_path, sampling_rate=rate, num_coeff=coeff, verbose=True)
