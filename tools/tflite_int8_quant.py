import cv2
import time
import numpy as np
import tensorflow as tf

RESIZE_INPUT = 227
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

def representative_dataset_gen():
    for image_path in raw_image_paths:
        print('Path: {}'.format(image_path))
        image = cv2.imread(image_path)
        image = cv2.resize(image, (227, 227))
        image = np.expand_dims(image, 0).astype(np.float32)

        image = (image - image.mean(axis = (0,1,2), keepdims=True)) / image.std(axis = (0,1,2), keepdims=True)
        image = cv2.normalize(image, None, 0, 1, cv2.NORM_MINMAX)
        image = image.astype(np.float32)

        yield [image]

print("[ INFO ] Loading image paths for representative dataset")
with open('/src/age_representative_dataset.txt') as f:
    raw_image_paths = f.read().split('\n')[:-1]

#print("[ INFO ] Printing raw image paths:")
#print(raw_image_paths)

print("[ INFO ] Loading tensorflow saved model")
#converter = tf.compat.v1.lite.TFLiteConverter.from_saved_model('/src/saved_models/stripped/1', input_arrays=['tf_example'], input_shapes={'tf_example': [1, 227, 227, 3]}, output_arrays=['Softmax'])
converter = tf.compat.v1.lite.TFLiteConverter.from_saved_model('/src/saved_models/stripped_2/1', output_arrays=['Softmax'])
print("[ INFO ] Complete")

# Int 8 quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops= [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.float32
converter.experimental_new_quantizer = True
#converter.default_ranges_stats = [0.0, 1.0]
#converter.inference_type = tf.uint8
#input_arrays = converter.get_input_arrays()
#converter.quantized_input_stats = {input_arrays[0]:(0.0,1.0)}
converter.representative_dataset = representative_dataset_gen

print("[ INFO ] Converting to int8 tflite model")
tflite_model_int8 = converter.convert()
print("[ INFO ] Complete")

print("[ INFO ] Exporting int8 tflite model")
open("/src/age_classify_int8_quant.tflite", "wb").write(tflite_model_int8)
print("[ INFO ] Complete")

