import tensorflow as tf
gpus = tf.config.list_physical_devices("GPU")
print(f"GPUs available: {gpus}")
if len(gpus) > 0:
    print("GPU check passed")
    exit(0)
else:
    print("No GPU detected")
    exit(1)