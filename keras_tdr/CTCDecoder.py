import tensorflow as tf

# CTC Beam search algorithm to improve text recognition results
def CTCDecoder():
    def decoder(y_pred):
        input_shape = tf.keras.backend.shape(y_pred)
        input_length = tf.ones(shape=input_shape[0]) * tf.keras.backend.cast(
            input_shape[1], 'float32')
        unpadded = tf.keras.backend.ctc_decode(y_pred, input_length)[0][0]
        unpadded_shape = tf.keras.backend.shape(unpadded)
        padded = tf.pad(unpadded,
                        paddings=[[0, 0], [0, input_shape[1] - unpadded_shape[1]]],
                        constant_values=-1)
        return padded

    return tf.keras.layers.Lambda(decoder, name='decode')