import onnx
import os
import tf2onnx

from tensorflow.keras.models import load_model
loaded_keras_model = load_model('RTFERmodel.h5')

onnx_model, _ = tf2onnx.convert.from_keras(loaded_keras_model)

onnx.save(onnx_model, 'RTFERmodel.onnx')