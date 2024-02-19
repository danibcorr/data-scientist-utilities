# --------------------------------------------------------------------------------------------
# LIBRARIES
# --------------------------------------------------------------------------------------------


from tensorflow.keras import layers
from tensorflow import keras
from .SqueezeAndExcitation import SqueezeAndExcitation
import tensorflow as tf
import tensorflow_probability as tfp


# --------------------------------------------------------------------------------------------
# SETS THE SEED OF THE RANDOM NUMBER GENERATOR
# --------------------------------------------------------------------------------------------


kernel_initial = tf.keras.initializers.TruncatedNormal(stddev=0.2, seed = 42)
bias_initial = tf.keras.initializers.Constant(value=0)


# --------------------------------------------------------------------------------------------
# CLASS DEFINITIONS
# -------------------------------------------------------------------------------------------- 


class StochasticDepthResidual(layers.Layer):

    def __init__(self, rate=0.5, **kwargs):

        super().__init__(**kwargs)
        self.rate = rate
        self.survival_probability = 1.0 - self.rate

    def call(self, x, training=None):

        if len(x) != 2:

            raise ValueError(
                f"""Input must be a list of length 2. """
                f"""Got input with length={len(x)}."""
            )

        shortcut, residual = x

        b_l = keras.backend.random_bernoulli([], p=self.survival_probability)

        if training:

            return shortcut + b_l * residual

        else:

            return shortcut + self.survival_probability * residual

    def get_config(self):

        config = {"rate": self.rate}
        base_config = super().get_config()

        return dict(list(base_config.items()) + list(config.items()))


class ResidualBlock(layers.Layer):
    
    def __init__(self, name, num_filters, drop_prob=0.1, layer_scale_init_value=1e-6):
        
        super(ResidualBlock, self).__init__()
        
        # Parameters
        self.name_layer = name
        self.num_filters = num_filters
        self.drop_prob = drop_prob
        self.layer_scale_init_value = layer_scale_init_value
        
        # SE blocks
        self.se_block = SqueezeAndExcitation(name = self.name_layer + f"_se_input", num_filters = self.num_filters)

        # Feature extraction
        self.layers = tf.keras.Sequential([
            layers.Conv2D(self.num_filters, kernel_size=7, padding="same", groups=num_filters, kernel_initializer=kernel_initial,
                          bias_initializer=bias_initial, name = self.name_layer + "_conv2d_7"),
            layers.LayerNormalization(name = self.name_layer + "_layernorm"),
            layers.Conv2D(self.num_filters * 4, kernel_size=1, padding="valid", kernel_initializer=kernel_initial, 
                          bias_initializer=bias_initial, name = self.name_layer + "_conv2d_4"),
            layers.Activation('gelu', name = self.name_layer + "_activation"),
            layers.Conv2D(self.num_filters, kernel_size=1, padding="valid", kernel_initializer=kernel_initial, 
                          bias_initializer=bias_initial, name = self.name_layer + "_conv2d_output")
        ], name = f"Sequential_Residual_{self.name_layer}")
        
        self.layer_scale_gamma = None
        
        if self.layer_scale_init_value > 0:
        
            with tf.init_scope():
                
                self.layer_scale_gamma = tf.Variable(name = self.name_layer + "_gamma", initial_value=self.layer_scale_init_value * tf.ones((self.num_filters)))

        self.stochastic_depth = StochasticDepthResidual(self.drop_prob)

    def call(self, inputs):
        
        # Feature extraction inputs
        x = self.layers(inputs)

        if self.layer_scale_gamma is not None:
            
            x = x * self.layer_scale_gamma
        
        # SE blocks
        x = self.se_block(x)

        # Regularization
        x = self.stochastic_depth([inputs, x])
        
        return x