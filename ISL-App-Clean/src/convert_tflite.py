import tensorflow as tf
import numpy as np
from config import MODEL_PATH, MODEL_DIR, SEQUENCE_LENGTH, FEATURE_SIZE

output_path = str(MODEL_DIR / "isl_model.tflite")

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

print("\nConverting to TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Required for LSTM (uses TensorListReserve internally)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS,
]
converter._experimental_lower_tensor_list_ops = False

tflite_model = converter.convert()

with open(output_path, "wb") as f:
    f.write(tflite_model)

print(f"\nSaved to: {output_path}")
print(f"Size: {len(tflite_model) / 1024:.1f} KB")

# Verify it works
print("\nVerifying...")
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"Input shape : {input_details[0]['shape']}")
print(f"Output shape: {output_details[0]['shape']}")

dummy = np.zeros((1, SEQUENCE_LENGTH, FEATURE_SIZE), dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], dummy)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])
print(f"Dummy output: {output}")
print("\nConversion successful!")