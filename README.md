# TensorFlow GPU Docker Setup

This repository contains Docker configuration for running TensorFlow with GPU support. The setup includes optimizations for WSL2 environments and includes all necessary packages for data science and deep learning tasks.

## Prerequisites

- Docker installed
- NVIDIA GPU with compatible drivers
- NVIDIA Container Toolkit installed
- If using WSL2: Properly configured GPU passthrough

## Included Packages

The Docker image includes the following packages:
- TensorFlow with GPU support
- NumPy
- Pandas
- scikit-learn (including MinMaxScaler)
- Keras layers and optimizers

## Setup Instructions

### 1. Build the Docker Image

Save the Dockerfile content as `Dockerfile.gpu` in your project directory, then build the image:

```bash
docker build -t tensorflow-gpu-custom -f Dockerfile.gpu .
```

This command creates a Docker image named `tensorflow-gpu-custom` based on the content of `Dockerfile.gpu`.

**Note:** The GPU support is included in the base image and internal configurations. You don't need GPU flags during the build process.

### 2. Run the Container with GPU Access

To run the container with GPU support:

```bash
docker run --gpus all -it tensorflow-gpu-custom
```

The `--gpus all` flag is what enables GPU access from within the container.

### 3. Verifying GPU Access

Once inside the container, you can verify GPU access with:

```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```

If your GPU is properly detected, this will list available GPUs.

## Troubleshooting

### Common Issues

1. **GPU Not Detected**
   - Error message: `failed call to cuInit: UNKNOWN ERROR (34)`
   - Error message: `kernel driver does not appear to be running on this host`
   
   Solutions:
   - Check NVIDIA driver installation on host: `nvidia-smi`
   - Verify NVIDIA Container Toolkit: `dpkg -l | grep nvidia-container-toolkit`
   - Test with a simple CUDA container: `docker run --gpus all --rm nvidia/cuda:11.6.2-base-ubuntu20.04 nvidia-smi`

2. **WSL2-Specific Issues**
   - Make sure WSL2 is using the latest kernel with GPU support
   - Ensure the NVIDIA drivers for WSL are installed on Windows
   - Check your `.wslconfig` file (in Windows user directory):
     ```
     [wsl2]
     kernelCommandLine = systemd.unified_cgroup_hierarchy=0
     ```

3. **Container Exiting Immediately**
   - Use the `-it` flags for interactive mode
   - If needed, keep it running in background: `docker run --gpus all -d tensorflow-gpu-custom tail -f /dev/null`

4. **Large Container Size**
   - The image includes CUDA libraries which are large
   - The Dockerfile includes cleanup steps to minimize size where possible

## Docker Commands Quick Reference

- Build the image: `docker build -t tensorflow-gpu-custom -f Dockerfile.gpu .`
- Run with GPU access: `docker run --gpus all -it tensorflow-gpu-custom`
- Check existing images: `docker images`
- Check running containers: `docker ps`
- Execute commands in running container: `docker exec -it CONTAINER_ID bash`

## Sample Code

To test that everything is working correctly:

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Check for GPU
print("TensorFlow version:", tf.__version__)
print("GPU Available:", len(tf.config.list_physical_devices('GPU')) > 0)
print("GPUs:", tf.config.list_physical_devices('GPU'))

# Create a simple model
model = tf.keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(10,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Generate some fake data
X = np.random.random((1000, 10))
y = np.random.randint(2, size=(1000, 1))

# Train the model
model.fit(X, y, epochs=5, batch_size=32)
```

## Notes

- The Docker image is based on TensorFlow 2.11.0-gpu for optimal compatibility
- The container automatically checks for CUDA installation and installs it if missing
- Environment variables are properly set for CUDA paths
