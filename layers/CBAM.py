# --------------------------------------------------------------------------------------------
# LIBRARIES
# --------------------------------------------------------------------------------------------


from tensorflow.keras import layers
import tensorflow as tf
from ..utils.padding import CircularPad


# --------------------------------------------------------------------------------------------
# CLASS DEFINITIONS
# https://arxiv.org/pdf/1807.06521.pdf
# --------------------------------------------------------------------------------------------


class ChannelAttentionModule(layers.Layer):


    def __init__(self, ratio = 16):

        super(ChannelAttentionModule, self).__init__()

        self.ratio = ratio
        self.l1 = None
        self.l2 = None

        self.gap = layers.GlobalAveragePooling2D()
        self.gmp = layers.GlobalMaxPooling2D()

        self.activation = layers.Activation('sigmoid')


    def build(self, input_shape):

        channel = input_shape[-1]
        self.l1 = layers.Dense(channel // self.ratio, activation = 'gelu', use_bias = False)
        self.l2 = layers.Dense(channel, use_bias = False)


    def call(self, inputs):

        avepool = self.gap(inputs)
        a = self.l1(avepool)
        a = self.l2(a)

        maxpool = self.gmp(inputs)
        m = self.l1(maxpool)
        m = self.l2(m)

        concat = a + m
        concat = sef.activation(concat)

        return layers.Multiply()([inputs, concat])


class SpatialAttentionModule(layers.Layer):


    def __init__(self):

        super(SpatialAttentionModule, self).__init__()

        self.padding = CircularPad((3, 3, 3, 3))
        self.conv = layers.Conv2D(1, kernel_size = 7, activation = 'sigmoid')


    def call(self, inputs):

        avepool = tf.reduce_mean(inputs, axis = -1)
        avepool = tf.expand_dims(avepool, axis = -1)

        maxpool = tf.reduce_max(inputs, axis = -1)
        maxpool = tf.expand_dims(maxpool, axis = -1)

        concat = layers.Concatenate()([avepool, maxpool])

        conv = self.conv(self.padding(concat))

        return layers.Multiply()([inputs, conv])


class CBAM(layers.Layer):


    def __init__(self):

        super(CBAM, self).__init__()
        
        self.channel_attention = ChannelAttentionModule()
        self.spatial_attention = SpatialAttentionModule()


    def call(self, inputs):

        attention = self.channel_attention(inputs)
        attention = self.spatial_attention(attention)

        return attention
