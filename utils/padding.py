# --------------------------------------------------------------------------------------------
# LIBRARIES
# --------------------------------------------------------------------------------------------


import tensorflow as tf
from tensorflow.keras import layers


# --------------------------------------------------------------------------------------------
# CLASS DEFINITIONS
# --------------------------------------------------------------------------------------------


class CircularPad(layers.Layer):

    """
    A custom layer to apply circular padding to the input tensor.
    """

    def __init__(self, padding = (1, 1, 1, 1)):

        """
        Initializes the CircularPad layer.

        Args:
            padding (tuple): Tuple specifying the padding sizes in the order (top, bottom, left, right).
        """

        super(CircularPad, self).__init__()

        self.pad_sizes = padding

    def call(self, x):

        """
        Applies circular padding to the input tensor.

        Args:
            x (tf.Tensor): Input tensor.

        Returns:
            tf.Tensor: Padded tensor.
        """

        top_pad, bottom_pad, left_pad, right_pad = self.pad_sizes

        # Circular padding for height dimension
        height_pad = tf.concat([x[:, -bottom_pad:], x, x[:, :top_pad]], axis=1)

        # Circular padding for width dimension
        return tf.concat([height_pad[:, :, -right_pad:], height_pad, height_pad[:, :, :left_pad]], axis=2)