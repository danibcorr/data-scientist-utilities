# --------------------------------------------------------------------------------------------
# LIBRARIES
# --------------------------------------------------------------------------------------------


import numpy as np
from numba import njit


# --------------------------------------------------------------------------------------------
# DEFINITIONS OF FUNCTIONS
# --------------------------------------------------------------------------------------------


@njit
def flip_horizontal(data):

    return np.fliplr(data)


@njit
def flip_vertical(data):
    
    return np.flipud(data)


@njit
def flip_both(data):

    return np.flipud(np.fliplr(data))


@njit
def custom_roll(array, shift, axis):
    
    shape = array.shape
    result = np.empty(shape, dtype=array.dtype)
    
    if axis == 0:
        
        result[:shift, :] = array[-shift:, :]
        result[shift:, :] = array[:-shift, :]
        
    elif axis == 1:
        
        result[:, :shift] = array[:, -shift:]
        result[:, shift:] = array[:, :-shift]
    
    else:
    
        raise ValueError("Unsupported axis")
    
    return result


@njit
def horizontal_transition(image, num_displacements, shift_values):
    
    result = np.empty((num_displacements,) + image.shape, dtype=image.dtype)

    for i in range(num_displacements):
    
        result[i] = custom_roll(image, shift_values[i], axis=1)

    return result


@njit
def vertical_transition(image, num_displacements, shift_values):
    
    result = np.empty((num_displacements,) + image.shape, dtype=image.dtype)

    for i in range(num_displacements):
    
        result[i] = custom_roll(image, shift_values[i], axis=0)

    return result