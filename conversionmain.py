import tensorflow as tf
import tf2onnx
model = tf.keras.models.load_model("unet_model.h5", compile=False)
spec = (tf.TensorSpec((1, 16, 16, 1), tf.float32, name="input"),)
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)
with open("unet_model.onnx", "wb") as f:
    f.write(model_proto.SerializeToString())


# pip  install tensorflow 
# pip install tf2onnx