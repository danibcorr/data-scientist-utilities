# --------------------------------------------------------------------------------------------
# LIBRARIES
# --------------------------------------------------------------------------------------------


import tensorflow as tf
from tensorflow.keras import layers


# --------------------------------------------------------------------------------------------
# CLASS DEFINITIONS
# --------------------------------------------------------------------------------------------


class RandomCrop(layers.Layer):


    def __init__(self, min_height, max_height, width, needs_tobe_multiple = False, multiple = 1, training = True):

        super(RandomCrop, self).__init__()

        self.min_height = min_height
        self.max_height = max_height
        self.width = width
        self.training = training

        self.needs_tobe_multiple = needs_tobe_multiple
        self.multiple = multiple


    def call(self, inputs):

        if self.training:

            # Randomly decides whether to apply the crops or not
            apply_crop = tf.random.uniform(shape = (), minval = 0, maxval = 1) < 0.5

            if apply_crop:

                # Generates random width and height sizes within specified ranges
                height = tf.random.uniform(shape = (), minval = self.min_height, maxval = self.max_height, dtype = tf.int32)
                height = ((height + (self.multiple - 1)) // self.multiple) * self.multiple if self.needs_tobe_multiple else height

                # Apply random crops to the input
                cropped_inputs = tf.image.random_crop(inputs, (tf.shape(inputs)[0], height, self.width, tf.shape(inputs)[-1]))
                
                return cropped_inputs

            else:

                return inputs
        
        else:

            return inputs

    def set_training_mode(self, training):

        self.training = training


class RandomCropLayer(layers.Layer):

    def call(self, inputs, min_height, needs_tobe_multiple = False, multiple = 1):
        
        if tf.greater(tf.shape(inputs)[1], min_height):

            return RandomCrop(min_height = min_height, max_height = tf.shape(inputs)[1], width = tf.shape(inputs)[2],
                              needs_tobe_multiple = False, multiple = 1, training = True)(inputs)
        
        else:

            return inputs