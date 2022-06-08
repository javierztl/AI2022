import tf2onnx
import onnxruntime as rt
import tensorflow as tf
import tensorflow_addons as tfa

model = tf.keras.models.load_model('facemask_personid_classifier_fine_best.h5')
spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)
output_path = 'model2.onnx'
tf2onnx.convert.from_keras(model, input_signature=spec, opset=13, output_path=output_path)