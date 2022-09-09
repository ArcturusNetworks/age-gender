import cv2
import time
import numpy as np
import tensorflow as tf

RESIZE_INPUT = 227
MEAN = (0.485, 0.456, 0.406) 
STD = (0.229, 0.224, 0.225)

tf.config.threading.set_intra_op_parallelism_threads(5)
tf.config.threading.set_inter_op_parallelism_threads(5)

print("[ INFO ] Creating tflite interpreter")
interpreter = tf.compat.v1.lite.Interpreter('/src/age_classify_fp16.tflite')
print("Complete")

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']

print("[ INFO ] Input Details: ", input_details)
print("[ INFO ] Output Details: ", output_details)

# NOTE: Refer to 'preprocess_age()' in guess_optimized.py for correct
#       preprocessing of images using tensorflow standardization

img_loc = "/src/daniel.jpg"
print("[ INFO ] Reading image: ", img_loc) 
img = cv2.imread(img_loc)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.resize(img, (RESIZE_INPUT, RESIZE_INPUT))
img = np.expand_dims(img, 0).astype(np.float32)

# Int8 quatization experimentation
#input_scale, input_zero_point = input_details[0]["quantization"]
#print("[ INFO ] Quantization Input Scale: ", input_scale)
#print("[ INFO ] Quantization Input Zero Point: ", input_zero_point)
#img = (img - img.mean(axis = (0,1,2), keepdims=True)) / img.std(axis = (0,1,2), keepdims=True)
#img = cv2.normalize(img, None, 0, 1, cv2.NORM_MINMAX)
#img = img / input_scale + input_zero_point

print("[ INFO ] Image Mean: ", img.mean())
print("[ INFO ] Image Std Dev: ", img.std())
img /= 255.0
img -= MEAN
img /= STD
#img = np.expand_dims(img, 0).astype(np.float32)

input_data = img

print("[ INFO ] Allocating tensors")
interpreter.allocate_tensors()

print("[ INFO ] Setting input tensor")
interpreter.set_tensor(input_details[0]['index'], input_data)

print("[ INFO ] Invoking tflite interpreter")
t0 = time.time(); interpreter.invoke(); t1 = time.time(); t_ms = (t1 - t0) * 100; print(f'Age Classification: {t_ms} ms')
t0 = time.time(); interpreter.invoke(); t1 = time.time(); t_ms = (t1 - t0) * 100; print(f'Age Classification: {t_ms} ms')
t0 = time.time(); interpreter.invoke(); t1 = time.time(); t_ms = (t1 - t0) * 100; print(f'Age Classification: {t_ms} ms')
t0 = time.time(); interpreter.invoke(); t1 = time.time(); t_ms = (t1 - t0) * 100; print(f'Age Classification: {t_ms} ms')
t0 = time.time(); interpreter.invoke(); t1 = time.time(); t_ms = (t1 - t0) * 100; print(f'Age Classification: {t_ms} ms')

print("[ INFO ] Extracting output tensor")
output_data = interpreter.get_tensor(output_details[0]['index'])

print("[ INFO ] Output Data: ", output_data)

