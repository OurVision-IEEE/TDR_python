import string

DEFAULT_BUILD_PARAMS = {
    'height': 31,
    'width': 200,
    'color': False,
    'filters': (64, 128, 256, 256, 512, 512, 512),
    'rnn_units': (128, 128),
    'dropout': 0.25,
    'rnn_steps_to_discard': 2,
    'pool_size': 2,
    'stn': True,
}

DEFAULT_ALPHABET = string.digits + string.ascii_lowercase

PRETRAINED_WEIGHTS = {
    'kurapan': {
        'alphabet': DEFAULT_ALPHABET,
        'build_params': DEFAULT_BUILD_PARAMS,
        'weights': {
            'notop': {
                'url':
                'https://github.com/faustomorales/keras-ocr/releases/download/v0.8.4/crnn_kurapan_notop.h5',
                'filename': 'crnn_kurapan_notop.h5',
                'sha256': '027fd2cced3cbea0c4f5894bb8e9e85bac04f11daf96b8fdcf1e4ee95dcf51b9'
            },
            'top': {
                'url':
                'https://github.com/faustomorales/keras-ocr/releases/download/v0.8.4/crnn_kurapan.h5',
                'filename': 'crnn_kurapan.h5',
                'sha256': 'a7d8086ac8f5c3d6a0a828f7d6fbabcaf815415dd125c32533013f85603be46d'
            }
        }
    }
}

dataset_path = './represent_data'