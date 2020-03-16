import os
from feature_extraction.spectral import generate_mfcc
from refactoring.transcript import index_transcripts


if __name__ == '__main__':

    data_path = 'F:\\Speech_Recognition_Macedonian\\Database\\edit'
    rate = 16000
    coeff = 13

    # mfcc_dat = np.load(data_path + os.sep + '1' + os.sep + '000001' + os.sep + '1-000001_mfcc.npy')

    index_transcripts(path=data_path, verbose=False)
