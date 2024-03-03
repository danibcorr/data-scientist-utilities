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


class MaxMinImportance(layers.Layer):


    def __init__(self, name, **kwargs):

        super(MaxMinImportance, self).__init__(**kwargs)

        self.name_layer = name

        self.p0 = self.add_weight(self.name_layer + "p0", shape = (), initializer = 'ones', trainable = True, constraint = lambda x: tf.clip_by_value(x, 0, 1))
        self.p1 = self.add_weight(self.name_layer + "p1", shape = (), initializer = 'ones', trainable = True, constraint = lambda x: tf.clip_by_value(x, 0, 1))


    def call(self, inputs):

        maxpool, minpool = inputs

        lambda_val = self.p0 ** 2 / (self.p0 ** 2 + self.p1 ** 2)
        one_minus_lambda = self.p1 ** 2 / (self.p0 ** 2 + self.p1 ** 2)

        return lambda_val * maxpool + one_minus_lambda * minpool


class GlobalMinPooling2D(layers.Layer):


    def __init__(self, **kwargs):

        super(GlobalMinPooling2D, self).__init__(**kwargs)


    def call(self, inputs):

        return tf.reduce_min(inputs, axis = [1, 2])


class ChannelAttentionModule(layers.Layer):


    def __init__(self, name, use_min = False, ratio = 16):

        super(ChannelAttentionModule, self).__init__()

        self.name_layer = name

        self.ratio = ratio

        self.l1 = None
        self.l2 = None

        self.variable = GlobalMinPooling2D(name = f"GMinP_CAM_{name}") if use_min else layers.GlobalAveragePooling2D(name = f"GAveP_CAM_{name}")
        self.gmp = layers.GlobalMaxPooling2D(name = f"GMaxP_CAM_{name}")
        self.mmi = MaxMinImportance(name = f"MMI_CAM_{name}")

        self.activation = layers.Activation('sigmoid', name = f"Activation_CAM_{name}")


    def build(self, input_shape):

        channel = input_shape[-1]

        self.l1 = layers.Dense(channel // self.ratio, activation = 'relu', use_bias = False)
        self.l2 = layers.Dense(channel, use_bias = False)


    def call(self, inputs):

        variable_pool = self.l2(self.l1(self.variable(inputs)))

        maxpool = self.l2(self.l1(self.gmp(inputs)))

        concat = self.activation(self.mmi([maxpool, variable_pool]))

        return layers.Multiply()([inputs, concat])


class SpatialAttentionModule(layers.Layer):


    def __init__(self, name, use_min = False):

        super(SpatialAttentionModule, self).__init__()

        self.name_layer = name

        self.use_min = use_min

        self.padding = CircularPad((3, 3, 3, 3))
        self.conv = layers.Conv2D(1, kernel_size = 7, activation = 'sigmoid', name = f"Conv2D_SAM_{name}")


    def call(self, inputs):

        variable_pool = tf.reduce_min(inputs, axis = -1) if use_min else tf.reduce_mean(inputs, axis = -1)
        variable_pool = tf.expand_dims(variable_pool, axis = -1)

        maxpool = tf.reduce_max(inputs, axis = -1)
        maxpool = tf.expand_dims(maxpool, axis = -1)

        concat = layers.Concatenate()([variable_pool, maxpool])

        conv = self.conv(self.padding(concat))

        return layers.Multiply()([inputs, conv])


class CBAM(layers.Layer):


    def __init__(self, name, use_min):

        super(CBAM, self).__init__()
        
        self.channel_attention = ChannelAttentionModule(name, use_min)
        self.spatial_attention = SpatialAttentionModule(name, use_min)


    def call(self, inputs):

        return self.spatial_attention(self.channel_attention(inputs))
