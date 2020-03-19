"""
Functions for preprocessing of generated spectral features

Copyright 2020 by Blagoj Hristov

See the LICENSE file for the licensing associated with this software.

Author:
  Blagoj Hristov, March 2020

"""

import numpy as np


def normalize(data):
    """
    This function is for normalization of the generated spectrogram or MFCC features.

    Parameters:
        data (np.array): NumPy array containing the spectrogram or MFCC features of a single audio file

    Returns:
        data (np.array): NumPy array containing the normalized data

    """

    mean = np.mean(data)
    std = np.std(data)
    data = (data - mean) / std

    return data


def padding(data, maximum):
    """
    This function is for zero-padding the feature array, to allow for shape uniformity during training.

    Parameters:
        data (np.array): NumPy array containing the spectrogram or MFCC features of a single audio file
        maximum (int): The maximum size (longest audio file) of the spectrogram or MFCC features, which determines the size of the padding.

    Returns:
        data (np.array): NumPy array containing the padded data

    """

    data = np.pad(data, ((0, maximum - len(data)), (0, 0)), 'constant', constant_values=0.)

    return data
