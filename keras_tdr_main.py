from keras_tdr.representative_data_gen import representative_data_gen
from keras_tdr.build_model import build_model
from keras_tdr.hyperparameters import DEFAULT_ALPHABET, DEFAULT_BUILD_PARAMS, PRETRAINED_WEIGHTS
from keras_tdr.network import download_and_verify
from keras_tdr.convert_tflite import convert_tflite
from keras_tdr.run_tflite_model import run_tflite_model

import cv2

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

    model.summary()

    dataset_path = './represent_data'
    
    quantization = 'dr' #@param ["dr", "float16"]
    convert_tflite(quantization, prediction_model)
    
    quantization = 'float16' #@param ["dr", "float16"]
    convert_tflite(quantization, prediction_model)

    quantization = 'int8'  #@param ["dr", "float16", 'int8', 'full_int8']
    convert_tflite(quantization, prediction_model)

    image_path = 'images/available.png'

    # Running Dynamic Range Quantization
    tflite_output = run_tflite_model(image_path, 'dr')
    final_output = "".join(alphabets[index] for index in tflite_output[0] if index not in [blank_index, -1])
    print(final_output)
    cv2.imshow(cv2.imread(image_path))

    # Running Float16 Quantization
    tflite_output = run_tflite_model(image_path, 'float16')
    final_output = "".join(alphabets[index] for index in tflite_output[0] if index not in [blank_index, -1])
    print(final_output)
    cv2.imshow(cv2.imread(image_path))

    # Running Integer Quantization
    tflite_output = run_tflite_model(image_path, 'int8')
    final_output = "".join(alphabets[index] for index in tflite_output[0] if index not in [blank_index, -1])
    print(final_output)
    cv2.imshow(cv2.imread(image_path))