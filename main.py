from tdr.build_model import build_model
from tdr.hyperparameters import DEFAULT_ALPHABET, DEFAULT_BUILD_PARAMS, PRETRAINED_WEIGHTS

if __name__ == "__main__":
    build_params = DEFAULT_BUILD_PARAMS
    alphabets = DEFAULT_ALPHABET
    blank_index = len(alphabets)

    model, prediction_model = build_model(alphabet=alphabets, **build_params)