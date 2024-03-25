# --------------------------------------------------------------------------------------------
# LIBRARIES
# --------------------------------------------------------------------------------------------


import tensorflow as tf
from tensorflow.keras import layers
import numpy as np


# --------------------------------------------------------------------------------------------
# CLASS DEFINITIONS
# --------------------------------------------------------------------------------------------


class RandomCrop(layers.Layer):

    """
    Randomly crops the input tensor.

    Args:
        min_height (int): Minimum height of the crop.
        max_height (int): Maximum height of the crop.
        width (int): Width of the crop.
        needs_tobe_multiple (bool): If True, ensures the cropped height is a multiple of 'multiple'.
        multiple (int): Value to ensure the cropped height is a multiple of.
        training (bool): Whether the layer is used in training mode or not.
    """

    def __init__(self, min_height, max_height, width, needs_tobe_multiple=False, multiple=1, training=True):

        super(RandomCrop, self).__init__()

        self.min_height = min_height
        self.max_height = max_height
        self.width = width
        self.training = training
        self.needs_tobe_multiple = needs_tobe_multiple
        self.multiple = multiple


    def call(self, inputs):

        if self.training:

            # Randomly decides whether to apply the crop or not
            apply_crop = tf.random.uniform(shape=(), minval=0, maxval=1) < 0.5

            if apply_crop:

                # Generates random height within specified range
                height = tf.random.uniform(shape=(), minval=self.min_height, maxval=self.max_height, dtype=tf.int32)
                height = ((height + (self.multiple - 1)) // self.multiple) * self.multiple if self.needs_tobe_multiple else height
                # Apply random crop to the input
                cropped_inputs = tf.image.random_crop(inputs, (tf.shape(inputs)[0], height, self.width, tf.shape(inputs)[-1]))

                return cropped_inputs

            else:

                return inputs
        else:

            return inputs


    def set_training_mode(self, training):

        """
        Sets the training mode of the layer.

        Args:
            training (bool): Whether the layer is used in training mode or not.
        """

        self.training = training


class RandomCropLayer(layers.Layer):

    """
    Applies random crop to input images based on the specified minimum height.

    Args:
        min_height (int): Minimum height for the random crop.
        needs_tobe_multiple (bool): If True, ensures the cropped height is a multiple of 'multiple'.
        multiple (int): Value to ensure the cropped height is a multiple of.
    """

    def call(self, inputs, min_height, needs_tobe_multiple=False, multiple=1):

        if tf.greater(tf.shape(inputs)[1], min_height):

            return RandomCrop(min_height=min_height, max_height=tf.shape(inputs)[1], width=tf.shape(inputs)[2],
                              needs_tobe_multiple=False, multiple=1, training=True)(inputs)
        else:

            return inputs


def RandomCropNumpy(data, multiple, min_height, width=24):

    """
    Randomly crops the input numpy array.

    Args:
        data (numpy.ndarray): Input numpy array.
        multiple (int): Value to ensure the cropped height is a multiple of.
        min_height (int): Minimum height for the random crop.
        width (int): Width of the crop.

    Returns:
        numpy.ndarray: Randomly cropped numpy array.
    """

    # Randomly decides whether to apply the crop or not, and if the minimum size is met.
    if np.random.uniform(0, 1) > 0.5 and data.shape[1] > min_height:

        # Generates random stop sizes within specified ranges
        height = np.random.randint(min_height, data.shape[1])
        height = ((height + (multiple - 1)) // multiple) * multiple

        # Applies random cropping to the input
        return tf.image.random_crop(data, (data.shape[0], height, width, data.shape[-1])).numpy()

    else:

        return data