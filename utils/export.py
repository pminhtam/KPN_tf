import tensorflow as tf
from model.kpn import kpn

# Create a model using high-level tf.keras.* APIs
input_tensor = tf.keras.Input(shape=[1024, 1024, 3])
output_tensor = kpn(input_tensor)
model = tf.keras.Model(input_tensor, output_tensor)
# Convert the model.

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
converter.experimental_new_converter = True

# Save the model.
with open('model_kpn_tf.tflite', 'wb') as f:
  f.write(tflite_model)