# TensorFlow GPU Docker Troubleshooting Guide

This guide helps you diagnose and resolve common issues with the TensorFlow GPU Docker setup.

## 🔍 Quick Diagnostic Steps

### Step 1: Check Your System

Run this command to check your system status:

```bash
# Check NVIDIA driver
nvidia-smi

# Check Docker
docker --version

# Check Docker Compose
docker-compose --version
```

**Expected Output:**

- `nvidia-smi` should show your GPU(s) and driver version
- Docker version should be 20.10+
- Docker Compose version should be 1.28+

### Step 2: Verify NVIDIA Container Toolkit

```bash
# Test NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:11.6-base-ubuntu20.04 nvidia-smi
```

**Expected Output:** Should display the same GPU information as your host system.

---

## 🚨 Common Issues & Solutions

### Issue: "No GPU devices found" or "Could not find CUDA"

**Symptoms:**

- Benchmark shows "Running on CPU only"
- TensorFlow doesn't detect GPU
- Error: `Could not load dynamic library 'libcudart.so'`

**Troubleshooting Flowchart:**

```
Start: GPU not detected
    ↓
[1] Can you run `nvidia-smi`?
    ├─ No → Install/Update NVIDIA drivers
    │       ↓
    │   [2] Driver version ≥ 450.80?
    │       ├─ No → Update to latest driver
    │       └─ Yes → Continue to [3]
    └─ Yes → Continue to [3]
    
[3] Can you run `docker run --rm --gpus all nvidia/cuda:11.6-base-ubuntu20.04 nvidia-smi`?
    ├─ No → Install nvidia-container-toolkit
    │       ↓
    │   [4] Follow installation steps below
    └─ Yes → Continue to [5]
    
[5] Does TensorFlow detect GPU in container?
    ├─ No → Check CUDA/TensorFlow compatibility
    │       ↓
    │   [6] Verify versions match support matrix
    └─ Yes → GPU setup successful! 🎉
```

**Solutions:**

**1. Install/Update NVIDIA Drivers**

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install nvidia-driver-525  # or latest version

# CentOS/RHEL
sudo dnf install nvidia-driver

# Verify installation
nvidia-smi
```

**2. Install NVIDIA Container Toolkit**

```bash
# Add NVIDIA repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install nvidia-container-toolkit
sudo apt update
sudo apt install nvidia-container-toolkit

# Restart Docker
sudo systemctl restart docker
```

**3. Check CUDA/TensorFlow Compatibility**

| TensorFlow Version | CUDA Version | cuDNN Version |
|-------------------|--------------|---------------|
| 2.13.0            | 11.8         | 8.6           |
| 2.12.0            | 11.8         | 8.6           |
| 2.11.0            | 11.2         | 8.1           |
| 2.10.0            | 11.2         | 8.1           |

---

### Issue: "Container fails to start" or "Permission denied"

**Symptoms:**

- Docker container exits immediately
- Permission errors accessing GPU
- `docker-compose up` fails

**Diagnosis Steps:**

```bash
# Check Docker daemon status
sudo systemctl status docker

# Check container logs
docker-compose logs tensorflow-gpu

# Test with minimal GPU container
docker run --rm --gpus all nvidia/cuda:11.6-runtime-ubuntu20.04 nvidia-smi
```

**Solutions:**

**1. Fix Docker Permissions**

```bash
# Add user to docker group
sudo usermod -aG docker $USER

# Log out and back in, or run:
newgrp docker
```

**2. Check GPU Access Permissions**

```bash
# Check GPU device permissions
ls -la /dev/nvidia*

# If needed, fix permissions
sudo chmod 666 /dev/nvidia*
```

---

### Issue: "Out of Memory" or Poor Performance

**Symptoms:**

- Benchmarks crash with OOM errors
- Very slow performance
- GPU memory errors

**Diagnosis Steps:**

```bash
# Monitor GPU memory usage
watch -n 1 nvidia-smi

# Check available memory
docker run --rm --gpus all tensorflow/tensorflow:2.11.0-gpu python -c "
import tensorflow as tf
print('GPU Memory Info:')
for gpu in tf.config.experimental.list_physical_devices('GPU'):
    print(f'GPU: {gpu}')
    details = tf.config.experimental.get_memory_info(gpu.name)
    print(f'Current: {details[\"current\"]/1024/1024:.0f}MB')
    print(f'Peak: {details[\"peak\"]/1024/1024:.0f}MB')
"
```

**Solutions:**

**1. Enable GPU Memory Growth**

```python
# In your TensorFlow code
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
```

**2. Reduce Batch Sizes**

```bash
# Run benchmarks with smaller batches
docker-compose run --rm tensorflow-gpu python /app/tf_benchmark.py --batch-size 512
```

**3. Set GPU Memory Limit**

```python
# Limit GPU memory usage
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]  # 4GB limit
        )
    except RuntimeError as e:
        print(e)
```

---

### Issue: "Image pull errors" or "Build failures"

**Symptoms:**

- Cannot pull TensorFlow images
- Docker build fails
- Network timeout errors

**Solutions:**

**1. Check Internet Connection & Docker Hub Access**

```bash
# Test connectivity
ping docker.io

# Try pulling base image manually
docker pull tensorflow/tensorflow:2.11.0-gpu
```

**2. Use Alternative Registry**

```bash
# Use NVIDIA NGC registry
docker pull nvcr.io/nvidia/tensorflow:22.12-tf2-py3
```

**3. Build with Different Base Image**

```dockerfile
# In Dockerfile, try different base images
FROM nvcr.io/nvidia/tensorflow:22.12-tf2-py3
# OR
FROM tensorflow/tensorflow:2.11.0-gpu-jupyter
```

---

## 🛠️ Advanced Diagnostics

### Verbose GPU Information

```bash
# Detailed GPU info
docker run --rm --gpus all tensorflow/tensorflow:2.11.0-gpu python -c "
import tensorflow as tf
print('TensorFlow Version:', tf.__version__)
print('CUDA Available:', tf.test.is_built_with_cuda())
print('GPU Available:', tf.test.is_gpu_available())
print('GPU Devices:', tf.config.list_physical_devices('GPU'))

# Test computation
with tf.device('/GPU:0'):
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
    c = tf.matmul(a, b)
    print('Test computation successful:', c.numpy())
"
```

### Container Environment Check

```bash
# Run interactive container for debugging
docker-compose run --rm tensorflow-gpu bash

# Inside container:
nvidia-smi
python -c "import tensorflow as tf; print(tf.config.list_physical_devices())"
```

---

## 📞 Getting Help

If none of these solutions work:

1. **Check the GitHub Issues**: [Project Issues](https://github.com/mrgkanev/tensorflow-gpu-docker/issues)

2. **Create a New Issue** with:
   - Your system specs (OS, GPU model, driver version)
   - Complete error logs
   - Output of diagnostic commands above

3. **Include Environment Details**:

   ```bash
   # Generate environment report
   docker run --rm --gpus all tensorflow/tensorflow:2.11.0-gpu python -c "
   import tensorflow as tf
   import sys
   print('Python:', sys.version)
   print('TensorFlow:', tf.__version__)
   print('CUDA:', tf.test.is_built_with_cuda())
   print('GPU Available:', tf.test.is_gpu_available())
   print('GPU Devices:', len(tf.config.list_physical_devices('GPU')))
   "
   ```

4. **Community Resources**:
   - [TensorFlow GPU Support Guide](https://www.tensorflow.org/install/gpu)
   - [NVIDIA Docker Documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
   - [Docker GPU Support](https://docs.docker.com/config/containers/resource_constraints/#gpu)

---

## ✅ Prevention Tips

- **Keep drivers updated**: Check for NVIDIA driver updates monthly
- **Monitor GPU temperature**: Ensure adequate cooling
- **Regular container updates**: Pull latest TensorFlow images periodically
- **Backup working configurations**: Save working docker-compose.yml versions
- **Test after system updates**: Verify GPU access after OS/driver updates
