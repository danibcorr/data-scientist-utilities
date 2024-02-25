# --------------------------------------------------------------------------------------------
# LIBRARIES
# --------------------------------------------------------------------------------------------


import tensorflow as tf
from tensorflow.keras import layers
from ..utils.padding import CircularPad


# --------------------------------------------------------------------------------------------
# CLASS DEFINITIONS
# https://github.com/nicholausdy/Adaptive-Polyphase-Sampling-Keras
# https://arxiv.org/pdf/2011.14214.pdf
# --------------------------------------------------------------------------------------------


class APSLayer(layers.Layer):


    def __init__(self, stride = 2, order = 2, name = None, **kwargs):

        super(APSLayer, self).__init__(name = name, **kwargs)

        self.stride = stride
        self.order = order


    def call(self, inputs):

        downsampled, max_norm_index = self.downsample(inputs)

        return downsampled, max_norm_index


    def get_config(self):

        config = super().get_config()
        config.update({"stride": self.stride, "order": self.order})

        return config


    @tf.function
    def downsample(self, inputs):
        
        polyphase_components = tf.TensorArray(tf.float32, size = 0, dynamic_size = True, clear_after_read = False)
        
        input_shape = tf.shape(inputs)
        arr_index = 0
        
        for i in range(self.stride):
        
            for j in range(self.stride):
        
                strided_matrix = tf.strided_slice(
                    inputs, 
                    begin = [0, i, j, 0], 
                    end = [0, input_shape[1], input_shape[2], input_shape[3]], 
                    strides = [1, self.stride, self.stride, 1],
                    begin_mask = 9,
                    end_mask = 9
                )
        
                strided_matrix = tf.cast(strided_matrix, dtype = tf.float32)
                polyphase_components = polyphase_components.write(arr_index, strided_matrix)
                arr_index += 1

        norms = tf.map_fn(
            lambda x: tf.norm(tensor = polyphase_components.read(x), ord = self.order),
            tf.range(self.stride * 2),
            fn_output_signature = tf.float32
        )

        max_norm_index = tf.math.argmax(norms)
        max_norm_index = tf.cast(max_norm_index, dtype = tf.int32)

        return polyphase_components.read(max_norm_index), max_norm_index


class APSDownsampleGivenPolyIndices(layers.Layer):


    def __init__(self, stride = 2, name=None, **kwargs):

        super(APSDownsampleGivenPolyIndices, self).__init__(name = name, **kwargs)

        self.stride = stride


    def call(self, inputs, max_poly_indices):   

        strided_matrix = self.downsample(inputs, max_poly_indices)

        return strided_matrix


    @tf.function(jit_compile = True)
    def downsample(self, inputs, max_poly_indices):

        i, j = tf.meshgrid(tf.range(self.stride), tf.range(self.stride), indexing = 'ij')

        elem = tf.stack([tf.zeros_like(i), i, j, tf.zeros_like(i)], axis = -1)
        elem = tf.reshape(elem, [-1, 4])

        lookup = tf.TensorArray(tf.int32, size = self.stride ** 2)
        lookup = lookup.unstack(elem)

        max_poly_indices = lookup.read(max_poly_indices)

        input_shape = tf.shape(inputs)

        strided_matrix = tf.strided_slice(
            inputs,
            begin=max_poly_indices,
            end=[0, input_shape[1], input_shape[2], input_shape[3]],
            strides=[1, self.stride, self.stride, 1],
            begin_mask = 9,
            end_mask = 9
        )

        return tf.cast(strided_matrix, dtype = tf.float32)


    def get_config(self):

        config = super().get_config()
        config.update({"stride": self.stride})
        
        return config


class APSDownsampling(layers.Layer):


    def __init__(self, filtros):
        
        super(APSDownsampling, self).__init__()

        self.padding = CircularPad((1, 1, 1, 1))

        self.conv_y = layers.Conv2D(kernel_size = 3, strides = 1, filters = filtros)
        self.aps_layer = APSLayer()
        self.norm_y = layers.LayerNormalization()
        self.activation_y = layers.Activation('gelu')

        self.downsampling = APSDownsampleGivenPolyIndices()
        self.conv_x = layers.Conv2D(kernel_size = 3, strides = 1, filters = filtros)
        self.add = layers.Add()
        self.norm_x = layers.LayerNormalization()
        self.activation_x = layers.Activation('gelu')


    def call(self, inputs):

        y, max_norm_index = self.aps_layer(inputs)
        y = self.norm_y(y)
        y = self.activation_y(y)
        y = self.conv_y(self.padding(y))

        x = self.downsampling(inputs, max_norm_index)
        x = self.conv_x(self.padding(x))

        x = self.add([x, y])
        x = self.norm_x(x)
        x = self.activation_x(x)

        return x