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

    """
    Performs horizontal flipping of the input data.

    Args:
        data (numpy.ndarray): Input data.

    Returns:
        numpy.ndarray: Horizontally flipped data.
    """

    return np.fliplr(data)


@njit
def flip_vertical(data):

    """
    Performs vertical flipping of the input data.

    Args:
        data (numpy.ndarray): Input data.

    Returns:
        numpy.ndarray: Vertically flipped data.
    """

    return np.flipud(data)


@njit
def flip_both(data):

    """
    Performs flipping of the input data both horizontally and vertically.

    Args:
        data (numpy.ndarray): Input data.

    Returns:
        numpy.ndarray: Horizontally and vertically flipped data.
    """

    return np.flipud(np.fliplr(data))


@njit
def custom_roll(array, shift, axis):

    """
    Rolls the elements of the array along a specified axis.

    Args:
        array (numpy.ndarray): Input array.
        shift (int): Number of places by which elements are shifted.
        axis (int): Axis along which to roll the elements.

    Returns:
        numpy.ndarray: Rolled array.
    """

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

    """
    Performs horizontal transition on the input image.

    Args:
        image (numpy.ndarray): Input image.
        num_displacements (int): Number of displacements to apply.
        shift_values (numpy.ndarray): Array of shift values.

    Returns:
        numpy.ndarray: Array of horizontally transitioned images.
    """

    result = np.empty((num_displacements,) + image.shape, dtype=image.dtype)

    for i in range(num_displacements):

        result[i] = custom_roll(image, shift_values[i], axis=1)

    return result


@njit
def vertical_transition(image, num_displacements, shift_values):

    """
    Performs vertical transition on the input image.

    Args:
        image (numpy.ndarray): Input image.
        num_displacements (int): Number of displacements to apply.
        shift_values (numpy.ndarray): Array of shift values.

    Returns:
        numpy.ndarray: Array of vertically transitioned images.
    """

    result = np.empty((num_displacements,) + image.shape, dtype=image.dtype)

    for i in range(num_displacements):

        result[i] = custom_roll(image, shift_values[i], axis=0)

    return result