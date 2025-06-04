import tensorflow as tf
import os

# Display environment info
print("TensorFlow version:", tf.__version__)
print("Python version:", tf.python.platform.build_info.python_version)
print("CUDA Availability:", tf.test.is_built_with_cuda())

# Check GPU devices
gpus = tf.config.list_physical_devices('GPU')
print("GPUs:", gpus)

if len(gpus) > 0:
    print("\n✅ GPU IS AVAILABLE!")
    
    # Get more details about the GPU
    try:
        gpu_details = tf.config.experimental.get_device_details(gpus[0])
        print("GPU Details:", gpu_details)
    except:
        print("Could not get detailed GPU information")
        
    # Try to allocate a small tensor on GPU to verify it's working
    try:
        with tf.device('/device:GPU:0'):
            a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            b = tf.constant([[1.0, 1.0], [1.0, 1.0]])
            c = tf.matmul(a, b)
            print("Test GPU computation result:", c)
            print("✅ GPU computation successful!")
    except Exception as e:
        print("❌ GPU computation failed with error:", e)
else:
    print("\n❌ NO GPU FOUND!")
    print("\nTroubleshooting steps:")
    print("1. Ensure NVIDIA drivers are installed on the host machine")
    print("   Run 'nvidia-smi' on the host to verify")
    
    print("\n2. Ensure nvidia-container-toolkit is installed:")
    print("   sudo apt-get install -y nvidia-container-toolkit")
    
    print("\n3. Make sure you're running the container with:")
    print("   docker run --gpus all -it your-image-name")
    
    print("\n4. Check if NVIDIA driver is accessible from container:")
    if os.path.exists('/proc/driver/nvidia/version'):
        with open('/proc/driver/nvidia/version', 'r') as f:
            print("NVIDIA driver version found:", f.read().strip())
    else:
        print("NVIDIA driver not accessible in container (/proc/driver/nvidia/version not found)")
        
    print("\n5. Check CUDA environment variables:")
    print("CUDA_VISIBLE_DEVICES:", os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set'))
    print("LD_LIBRARY_PATH:", os.environ.get('LD_LIBRARY_PATH', 'Not set'))
