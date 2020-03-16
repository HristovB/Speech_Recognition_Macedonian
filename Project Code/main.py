import os
from feature_extraction.spectral import generate_mfcc
from refactoring.transcript import index_transcripts
from refactoring.audio_files import rename_files
from preprocessing.audio import resample_audio


if __name__ == '__main__':

    data_path = 'F:\\Speech_Recognition_Macedonian\\Database\\train_new'
    rate = 16000
    coeff = 13

    # mfcc_dat = np.load(data_path + os.sep + '1' + os.sep + '000001' + os.sep + '1-000001_mfcc.npy')

    resample_audio(path=data_path, sampling_rate=rate, verbose=True)
