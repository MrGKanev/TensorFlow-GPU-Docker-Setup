#!/usr/bin/env python3
"""
TensorFlow GPU Test Script

This script tests if TensorFlow can access the GPU and runs a simple
benchmark to verify performance.
"""

import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.optimizers.legacy import Adam

def print_separator(message):
    """Print a message with separators for better readability."""
    print("\n" + "=" * 70)
    print(f" {message}")
    print("=" * 70)

def check_environment():
    """Check TensorFlow and GPU environment."""
    print_separator("TensorFlow and GPU Environment Check")
    
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Python version: {tf.python.platform.build_info.python_version}")
    print(f"CUDA Availability: {tf.test.is_built_with_cuda()}")
    
    # Check GPU devices
    gpus = tf.config.list_physical_devices('GPU')
    print(f"GPUs: {gpus}")
    
    if len(gpus) > 0:
        print("\n✅ GPU IS AVAILABLE!")
        
        # Configure memory growth to avoid OOM errors
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                print(f"Memory growth enabled for: {gpu}")
        except RuntimeError as e:
            print(f"Memory growth configuration failed: {e}")
            print("This is normal if GPU is already initialized")
            
        # Get GPU details
        try:
            gpu_details = tf.config.experimental.get_device_details(gpus[0])
            print(f"GPU Details: {gpu_details}")
        except Exception as e:
            print(f"Could not get detailed GPU information: {e}")
            
        return True
    else:
        print("\n❌ NO GPU FOUND!")
        print_troubleshooting_steps()
        return False

def print_troubleshooting_steps():
    """Print troubleshooting steps when GPU is not detected."""
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
            print(f"NVIDIA driver version found: {f.read().strip()}")
    else:
        print("NVIDIA driver not accessible in container")
        
    print("\n5. Check CUDA environment variables:")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'Not set')}")

def test_gpu_computation():
    """Test basic GPU computation with matrix multiplication."""
    print_separator("GPU Computation Test")
    
    try:
        # Test GPU computation
        with tf.device('/device:GPU:0'):
            print("Creating test matrices...")
            a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            c = tf.matmul(a, b)
            print(f"Matrix multiplication result:\n{c}")
            print("✅ GPU computation successful!")
            return True
    except Exception as e:
        print(f"❌ GPU computation failed: {e}")
        return False

def benchmark_performance(matrix_size=2000, iterations=5):
    """Run a simple performance benchmark comparing CPU vs GPU."""
    print_separator("Performance Benchmark")
    
    print(f"Running benchmark with {matrix_size}x{matrix_size} matrices...")
    
    # Create test matrices
    a = tf.random.normal((matrix_size, matrix_size))
    b = tf.random.normal((matrix_size, matrix_size))
    
    # GPU benchmark
    try:
        with tf.device('/device:GPU:0'):
            print("Testing GPU performance...")
            start_time = time.time()
            for _ in range(iterations):
                c = tf.matmul(a, b)
            # Force execution
            _ = c.numpy()
            gpu_time = time.time() - start_time
            print(f"GPU time: {gpu_time:.3f} seconds ({gpu_time/iterations:.3f}s per iteration)")
    except Exception as e:
        print(f"GPU benchmark failed: {e}")
        gpu_time = None
    
    # CPU benchmark
    try:
        with tf.device('/device:CPU:0'):
            print("Testing CPU performance...")
            start_time = time.time()
            for _ in range(iterations):
                c = tf.matmul(a, b)
            # Force execution
            _ = c.numpy()
            cpu_time = time.time() - start_time
            print(f"CPU time: {cpu_time:.3f} seconds ({cpu_time/iterations:.3f}s per iteration)")
    except Exception as e:
        print(f"CPU benchmark failed: {e}")
        cpu_time = None
    
    # Compare performance
    if gpu_time and cpu_time:
        speedup = cpu_time / gpu_time
        print(f"\n🚀 GPU speedup: {speedup:.2f}x faster than CPU")
        if speedup > 1.5:
            print("✅ GPU is providing significant performance improvement!")
        else:
            print("⚠️  GPU speedup is lower than expected. Check GPU utilization.")
    else:
        print("❌ Could not compare CPU vs GPU performance")

def test_neural_network():
    """Test training a simple neural network on GPU."""
    print_separator("Neural Network Training Test")
    
    try:
        # Generate sample data
        print("Creating sample dataset...")
        X = np.random.random((1000, 20))
        y = np.random.randint(2, size=(1000, 1))
        
        # Create a simple model
        print("Building neural network...")
        model = tf.keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(20,)),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        
        # Compile model with legacy optimizer
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Train model
        print("Training model...")
        start_time = time.time()
        history = model.fit(
            X, y, 
            epochs=5, 
            batch_size=32, 
            validation_split=0.2,
            verbose=1
        )
        training_time = time.time() - start_time
        
        print(f"\n✅ Neural network training completed in {training_time:.2f} seconds")
        
        # Show final metrics
        final_loss = history.history['loss'][-1]
        final_acc = history.history['accuracy'][-1]
        print(f"Final training accuracy: {final_acc:.4f}")
        print(f"Final training loss: {final_loss:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Neural network training failed: {e}")
        return False

def main():
    """Run all GPU tests."""
    print("🚀 Starting TensorFlow GPU Test Suite")
    
    # Check environment
    has_gpu = check_environment()
    
    if not has_gpu:
        print("\n❌ GPU not available. Exiting test suite.")
        return
    
    # Run tests
    tests_passed = 0
    total_tests = 3
    
    # Test 1: Basic GPU computation
    if test_gpu_computation():
        tests_passed += 1
    
    # Test 2: Performance benchmark
    try:
        benchmark_performance()
        tests_passed += 1
    except Exception as e:
        print(f"Benchmark failed: {e}")
    
    # Test 3: Neural network training
    if test_neural_network():
        tests_passed += 1
    
    # Final summary
    print_separator("Test Summary")
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! GPU setup is working correctly.")
    elif tests_passed > 0:
        print("⚠️  Some tests passed. GPU is partially working.")
    else:
        print("❌ All tests failed. GPU setup needs troubleshooting.")

if __name__ == "__main__":
    main()