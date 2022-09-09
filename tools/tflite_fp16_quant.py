import cv2
import time
import numpy as np
import tensorflow as tf

print("[ INFO ] Loading tensorflow saved model")
converter = tf.compat.v1.lite.TFLiteConverter.from_saved_model('/src/saved_models/stripped/1', input_arrays=['tf_example'], input_shapes={'tf_example': [1, 227, 227, 3]}, output_arrays=['Softmax'])
#converter = tf.compat.v1.lite.TFLiteConverter.from_saved_model('/src/saved_models/stripped_2/1', output_arrays=['Softmax'])
print("[ INFO ] Complete")

# Float 16 post training quantization
#converter.allow_custom_ops = True
#converter.experimental_new_converter = True
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec_supported_types = [tf.float16]
#converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
#converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]

print("[ INFO ] Converting to fp16 tflite model")
tflite_model = converter.convert()
print("[ INFO ] Complete")

print("[ INFO ] Exporting fp16 tflite model")
open("/src/age_classify_fp16_quant.tflite", "wb").write(tflite_model)
print("[ INFO ] Complete")

