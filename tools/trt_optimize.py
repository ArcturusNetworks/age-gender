import numpy as np
import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt

print("[ INFO ] Loading tensorflow saved model into TrtGraph")
converter = trt.TrtGraphConverter(
        input_saved_model_dir='/src/saved_models/original/1',
        max_workspace_size_bytes=(11<<23),
        precision_mode="FP16",
        maximum_cached_engines=100)
print("[ INFO ] Complete")

print("[ INFO ] Converting trt graph")
converter.convert()
print("[ INFO ] Complete")

#print("[ INFO ] Building trt engine")
#input_data = np.random.random([1,227,227,3])
#converter.build([input_data])
#print("[ INFO ] Complete")

print("[ INFO ] Saving optimized trt graph")
converter.save('/src/saved_models/trt_orig/')
print("[ INFO ] Complete")

