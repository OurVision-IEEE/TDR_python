import tensorflow as tf
from tensorflow import keras

# Sigmoid Activation
def swish(x, beta=1):
    return x * keras.backend.sigmoid(beta * x)

def setup_swish():
    keras.utils.get_custom_objects().update({'swish': keras.layers.Activation(swish)})