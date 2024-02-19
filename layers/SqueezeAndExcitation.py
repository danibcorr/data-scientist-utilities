# --------------------------------------------------------------------------------------------
# LIBRARIES
# --------------------------------------------------------------------------------------------


from tensorflow.keras import layers
import tensorflow as tf


# --------------------------------------------------------------------------------------------
# CLASS DEFINITIONS
# --------------------------------------------------------------------------------------------


class SqueezeAndExcitation(layers.Layer):
    
    def __init__(self, name, num_filters, expansion = 0.25):
        
        super(SqueezeAndExcitation, self).__init__()
        
        self.name_layer = name
        
        self.layers = tf.keras.Sequential([
            layers.GlobalAvgPool2D(keepdims = True, name = self.name_layer + "_se_gap_2d"),
            layers.Dense(int(num_filters * expansion), use_bias=False, activation = 'gelu', name = self.name_layer + "_se_dense_gelu"),
            layers.Dense(num_filters, use_bias=False, activation = 'sigmoid', name = self.name_layer + "_se_dense_sigmoid")
        ])

    def call(self, inputs):
        
        x = self.layers(inputs)
        
        return x * inputs