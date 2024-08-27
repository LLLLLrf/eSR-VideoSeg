import onnx
from onnx2keras import onnx_to_keras
import tensorflow as tf

# 加载ONNX模型
onnx_model = onnx.load("model.onnx")

# 转换为Keras模型
keras_model = onnx_to_keras(onnx_model=onnx_model, input_names=["input_0"])

# 转换为TensorFlow Lite模型
converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
tflite_model = converter.convert()

# 保存为.tflite文件
with open("converted_model.tflite", "wb") as f:
    f.write(tflite_model)