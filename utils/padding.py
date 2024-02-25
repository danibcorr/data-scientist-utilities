# --------------------------------------------------------------------------------------------
# LIBRARIES
# --------------------------------------------------------------------------------------------


import tensorflow as tf
from tensorflow.keras import layers


# --------------------------------------------------------------------------------------------
# CLASS DEFINITIONS
# --------------------------------------------------------------------------------------------


class CircularPad(layers.Layer):
 

    def __init__(self, padding = (1, 1, 1, 1)):
 
        super(CircularPad, self).__init__()
        
        self.pad_sizes = padding


    def call(self, x):
 
        top_pad, bottom_pad, left_pad, right_pad = self.pad_sizes
        height_pad = tf.concat([x[:, -bottom_pad:], x, x[:, :top_pad]], axis = 1)
 
        return tf.concat([height_pad[:, :, -right_pad:], height_pad, height_pad[:, :, :left_pad]], axis = 2)