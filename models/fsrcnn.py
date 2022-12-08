import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Input, Rescaling


def fsrcnn(factor, channels=3):
    conv_args = {
        "activation": "relu",
        "kernel_initializer": "he_normal",
        "padding": "same",
    }
    # Input and normalize
    x_in = Input(shape=(None, None, channels))
    x = Rescaling(1./127.5, offset=-1)(x_in)

    x = Conv2D(56, 5, **conv_args)(x)
    x = Conv2D(16, 1, **conv_args)(x)
    x = Conv2D(12, 3, **conv_args)(x)
    x = Conv2D(12, 3, **conv_args)(x)
    x = Conv2D(12, 3, **conv_args)(x)
    x = Conv2D(12, 3, **conv_args)(x)
    x = Conv2D(56, 5, **conv_args)(x)
    x = Conv2D(channels * (factor ** 2), 3, **conv_args)(x)
    x = tf.nn.depth_to_space(x, factor)

    # Undo normalization and return model
    x = Rescaling(127.5, offset=127.5)(x)
    return Model(x_in, x, name="fsrcnn")
