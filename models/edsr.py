import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Add, Conv2D, Input, Lambda, Rescaling


# EDSR model https://arxiv.org/abs/1707.02921
def edsr(factor, filters=64, residual_blocks=16, residual_scaling=0.1, channels=3):
    # Input and normalize
    x_in = Input(shape=(None, None, channels))
    x = Rescaling(1./127.5, offset=-1)(x_in)

    # First convolutional and residual blocks combined
    x = r = Conv2D(filters, 3, padding='same')(x)
    for _ in range(residual_blocks):
        r = residual_block(r, filters, residual_scaling)
    r = Conv2D(filters, 3, padding='same')(r)
    x = Add()([x, r])

    # Last upsample and convolutional
    x = upsample(x, factor, filters)
    x = Conv2D(channels, 3, padding='same')(x)

    # Undo normalization and return model
    x = Rescaling(127.5, offset=127.5)(x)
    return Model(x_in, x, name="edsr")


# EDSR Residual block
# Add input and output from block (convolutional, ReLU, and convolutional).
def residual_block(x_in, filters, scaling):
    x = Conv2D(filters, 3, padding='same', activation='relu')(x_in)
    x = Conv2D(filters, 3, padding='same')(x)
    if scaling:
        x = Lambda(lambda v: v * scaling)(x)
    x = Add()([x_in, x])
    return x


# EDSR Upsample blocks
# Convolutional and Shuffle (different based on factor)
def upsample(x, factor, filters):
    def upsample_1(x, factor):
        x = Conv2D(filters * (factor ** 2), 3, padding='same')(x)
        return Lambda(lambda x: tf.nn.depth_to_space(x, factor))(x)

    if factor == 2:
        x = upsample_1(x, 2)
    elif factor == 3:
        x = upsample_1(x, 3)
    elif factor == 4:
        x = upsample_1(x, 2)
        x = upsample_1(x, 2)

    return x
