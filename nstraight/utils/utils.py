'''
Utility functions for Neural Straightening analysis
Author: Santiago Cadena
email: santiago.cadena@uni-tuebingen.de
'''

import numpy as np
from array2gif import write_gif
from scipy import signal
import hashlib
from skimage.transform import rescale


def key_hash(key):
    """
    32-byte hash used for lookup of primary keys of jobs
    """
    hashed = hashlib.md5()
    for k, v in sorted(key.items()):
        hashed.update(str(v).encode())
    return hashed.hexdigest()

def butter_temporal_filter(x, wn, fs, order=2, **kwargs):
    '''
    Filters the input array along it's first axis with a butterworth filter
    Args:
        :x:     input array
        :wn:    the cut-off frequency in Hz
        :fs:    input sampling frequency in Hz
        :order: int with the oder of the filer
    Returns
        :x_filtered: filtered array
    '''
    b, a       = signal.butter(N = order, Wn = wn, fs = fs) # get filter
    x_filtered = signal.filtfilt(b, a, x, axis=0)           # filter array
    return x_filtered
    
    
def subsample(x, step, **kwargs):
    '''
    Downsamples input movie via subsampling
    '''
    x = x[:, ::step, ::step]
    return x

def rescale_interpolation(x, scale, **kwargs):
    '''
    Downsamples input movie via interpolation
    '''
    x = rescale(x, [1, scale, scale], mode='reflect', \
                         multichannel=False, anti_aliasing=True, preserve_range=True)
    return x
    
def reshape_for_gif(x):
    '''
    For an input array shaped tsteps x height x width creates array compatible
    with the write_gif function
    '''
    return np.tile(x[None, ].transpose(1,2,3,0), 3).transpose(0,3,1,2)

def create_gif(x, file_path, fs):
    '''
    Creates a gif clip
    Args:
        :x: numpy array of size t_steps x height x width
        :file_path: absolute full path to store the gif
        :fs: The (integer) frames/second of the animatio
        :return: None
    '''
    if np.any(x > 255):
        x = x.astype(np.float64) / x.max() # normalize the data to 0 - 1
        x = 255 * x # Scale by 255
        x = x.astype(np.uint8)  # convert to uint8
    
    write_gif(reshape_for_gif(x), file_path, fs)

def get_trial_idx(dataset, trial):
    '''
    Finds the index in the dataset corresponding to a trial
    '''
    return np.where(dataset.trial_idx == trial)[0][0]


def type_object_movie(name):
    if 'bgv' in name:
        return 'type3'
    if 'v5' in name:
        return 'type1'
    if 'v6' in name:
        return 'type2' 