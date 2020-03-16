import os
from feature_extraction.spectral import generate_mfcc


if __name__ == '__main__':

    data_path = 'F:\\Speech_Recognition_Macedonian\\Database\\train'
    rate = 16000
    coeff = 13

    # mfcc_dat = np.load(data_path + os.sep + '1' + os.sep + '000001' + os.sep + '1-000001_mfcc.npy')

    for folder in os.listdir(data_path):
        print('Loading folder', folder, '...')

        generate_mfcc(path=data_path + os.sep + folder, sampling_rate=rate, num_coeff=coeff, folder_name=folder, verbose=True)

        print('Folder', folder, 'done!')
        print()
