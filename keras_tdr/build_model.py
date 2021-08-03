from tensorflow import keras
from .transform import _transform
from .CTCDecoder import CTCDecoder

import numpy as np

def build_model(alphabet,
                height,
                width,
                color,
                filters,
                rnn_units,
                dropout,
                rnn_steps_to_discard,
                pool_size,
                stn=True):
    """Build a Keras CRNN model for character recognition.
    Args:
        height: The height of cropped images
        width: The width of cropped images
        color: Whether the inputs should be in color (RGB)
        filters: The number of filters to use for each of the 7 convolutional layers
        rnn_units: The number of units for each of the RNN layers
        dropout: The dropout to use for the final layer
        rnn_steps_to_discard: The number of initial RNN steps to discard
        pool_size: The size of the pooling steps
        stn: Whether to add a Spatial Transformer layer
    """
    assert len(filters) == 7, '7 CNN filters must be provided.'
    assert len(rnn_units) == 2, '2 RNN filters must be provided.'
    inputs = keras.layers.Input((height, width, 3 if color else 1), name='input', batch_size=1)
    x = keras.layers.Permute((2, 1, 3))(inputs)
    x = keras.layers.Lambda(lambda x: x[:, :, ::-1])(x)
    x = keras.layers.Conv2D(filters[0], (3, 3), activation='relu', padding='same', name='conv_1')(x)
    x = keras.layers.Conv2D(filters[1], (3, 3), activation='relu', padding='same', name='conv_2')(x)
    x = keras.layers.Conv2D(filters[2], (3, 3), activation='relu', padding='same', name='conv_3')(x)
    x = keras.layers.BatchNormalization(name='bn_3')(x)
    x = keras.layers.MaxPooling2D(pool_size=(pool_size, pool_size), name='maxpool_3')(x)
    x = keras.layers.Conv2D(filters[3], (3, 3), activation='relu', padding='same', name='conv_4')(x)
    x = keras.layers.Conv2D(filters[4], (3, 3), activation='relu', padding='same', name='conv_5')(x)
    x = keras.layers.BatchNormalization(name='bn_5')(x)
    x = keras.layers.MaxPooling2D(pool_size=(pool_size, pool_size), name='maxpool_5')(x)
    x = keras.layers.Conv2D(filters[5], (3, 3), activation='relu', padding='same', name='conv_6')(x)
    x = keras.layers.Conv2D(filters[6], (3, 3), activation='relu', padding='same', name='conv_7')(x)
    x = keras.layers.BatchNormalization(name='bn_7')(x)
    if stn:
        # pylint: disable=pointless-string-statement
        """Spatial Transformer Layer
        Implements a spatial transformer layer as described in [1]_.
        Borrowed from [2]_:
        downsample_fator : float
            A value of 1 will keep the orignal size of the image.
            Values larger than 1 will down sample the image. Values below 1 will
            upsample the image.
            example image: height= 100, width = 200
            downsample_factor = 2
            output image will then be 50, 100
        References
        ----------
        .. [1]  Spatial Transformer Networks
                Max Jaderberg, Karen Simonyan, Andrew Zisserman, Koray Kavukcuoglu
                Submitted on 5 Jun 2015
        .. [2]  https://github.com/skaae/transformer_network/blob/master/transformerlayer.py
        .. [3]  https://github.com/EderSantana/seya/blob/keras1/seya/layers/attention.py
        """
        stn_input_output_shape = (width // pool_size**2, height // pool_size**2, filters[6])
        stn_input_layer = keras.layers.Input(shape=stn_input_output_shape)
        locnet_y = keras.layers.Conv2D(16, (5, 5), padding='same',
                                       activation='relu')(stn_input_layer)
        locnet_y = keras.layers.Conv2D(32, (5, 5), padding='same', activation='relu')(locnet_y)
        locnet_y = keras.layers.Flatten()(locnet_y)
        locnet_y = keras.layers.Dense(64, activation='relu')(locnet_y)
        locnet_y = keras.layers.Dense(6,
                                      weights=[
                                          np.zeros((64, 6), dtype='float32'),
                                          np.float32([[1, 0, 0], [0, 1, 0]]).flatten()
                                      ])(locnet_y)
        localization_net = keras.models.Model(inputs=stn_input_layer, outputs=locnet_y)
        x = keras.layers.Lambda(_transform,
                                output_shape=stn_input_output_shape)([x, localization_net(x)])
    x = keras.layers.Reshape(target_shape=(width // pool_size**2,
                                           (height // pool_size**2) * filters[-1]),
                             name='reshape')(x)

    x = keras.layers.Dense(rnn_units[0], activation='relu', name='fc_9')(x)

    rnn_1_forward = keras.layers.LSTM(rnn_units[0],
                                      kernel_initializer="he_normal",
                                      return_sequences=True,
                                      name='lstm_10')(x)
    rnn_1_back = keras.layers.LSTM(rnn_units[0],
                                   kernel_initializer="he_normal",
                                   go_backwards=True,
                                   return_sequences=True,
                                   name='lstm_10_back')(x)
    rnn_1_add = keras.layers.Add()([rnn_1_forward, rnn_1_back])
    rnn_2_forward = keras.layers.LSTM(rnn_units[1],
                                      kernel_initializer="he_normal",
                                      return_sequences=True,
                                      name='lstm_11')(rnn_1_add)
    rnn_2_back = keras.layers.LSTM(rnn_units[1],
                                   kernel_initializer="he_normal",
                                   go_backwards=True,
                                   return_sequences=True,
                                   name='lstm_11_back')(rnn_1_add)
    x = keras.layers.Concatenate()([rnn_2_forward, rnn_2_back])
    backbone = keras.models.Model(inputs=inputs, outputs=x)
    x = keras.layers.Dropout(dropout, name='dropout')(x)
    x = keras.layers.Dense(len(alphabet) + 1,
                           kernel_initializer='he_normal',
                           activation='softmax',
                           name='fc_12')(x)
    x = keras.layers.Lambda(lambda x: x[:, rnn_steps_to_discard:])(x)
    model = keras.models.Model(inputs=inputs, outputs=x)
    prediction_model = keras.models.Model(inputs=inputs, outputs=CTCDecoder()(model.output))
    return model, prediction_model