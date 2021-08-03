from tdr.build_model import build_model
from tdr.hyperparameters import DEFAULT_ALPHABET, DEFAULT_BUILD_PARAMS, PRETRAINED_WEIGHTS
from tdr.network import download_and_verify

if __name__ == "__main__":
    build_params = DEFAULT_BUILD_PARAMS
    alphabets = DEFAULT_ALPHABET
    blank_index = len(alphabets)

    model, prediction_model = build_model(alphabet=alphabets, **build_params)

    weights_dict = PRETRAINED_WEIGHTS['kurapan']

    model.load_weights(
        download_and_verify(
            url=weights_dict['weights']['top']['url'],
            filename=weights_dict['weights']['top']['filename'],
            sha256=weights_dict['weights']['top']['sha256']
        )
    )