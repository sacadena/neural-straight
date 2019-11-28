"""
Utility functions for curvature computations

Author: Santiago Cadena
Date:   11.19
email:  santiago.cadena@uni-tuebingen.de
"""
import numpy as np
from numpy.linalg import norm

def consecutive_difference(x):
    '''
    Computes the difference of consecutive elements along the first axis
    '''
    return np.diff(x, n=1, axis=0)

def normalize_arrays(x):
    '''
    Normalize each array along the first axis
    '''
    return np.array([a / norm(a.flatten()) for a in x])

def compute_curvature(x, deg=True):
    """
    Compute curvature by computing arccosine of consecutive normalized
    difference vectors [Hennaff et. al 2019]
    
    Args:
        x    : array with at least two dims.
        deg  : bool  True (default) for curvature output in degrees. Otherwise in rads.
    Returns:
        curvature : array with (1xM) with the curvature trajectory
    """
    
    x_diff   = consecutive_difference(x)
    x_diff_n = normalize_arrays(x_diff)
    
    # Flatten arrays along the first axis
    T      = x_diff_n.shape[0] 
    x_flat = x_diff_n.reshape(T, -1)
    
    # shifted arrays
    x0 = x_flat[:-1,]
    x1 = x_flat[1:,]
    
    cosine    = np.diag(np.dot(x0, x1.T)) # dot product consecutive vectors
    cosine    = [c if c <= 1 else 1 for c in cosine] # sanity check to clip values at 1
    
    # curvature computation
    curvature = np.arccos(cosine)
    
    if deg:
        curvature = np.degrees(curvature)
    
    return curvature
