#!/usr/bin/env python3
"""
TensorFlow GPU Quick Benchmark

A lightweight version of the benchmark for Docker health checks or quick testing.
"""

import os
import time
import json
import platform
import numpy as np
import tensorflow as tf
from datetime import datetime

# Ensure TensorFlow logs only errors, not warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def configure_gpu():
    """Configure GPUs to use memory growth and prevent OOM errors."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ GPU memory growth enabled on {len(gpus)} GPUs")
        except RuntimeError as e:
            print(f"❌ Error setting GPU memory growth: {e}")
    return gpus

def get_system_info():
    """Gather basic system information."""
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "tensorflow_version": tf.__version__,
        "is_built_with_cuda": tf.test.is_built_with_cuda(),
        "gpu_count": len(tf.config.list_physical_devices('GPU'))
    }

def run_matrix_multiplication_test(size=2048):
    """Run a simple matrix multiplication test."""
    print(f"Running {size}x{size} matrix multiplication test...")
    
    # Create random matrices
    a = tf.random.normal([size, size])
    b = tf.random.normal([size, size])
    
    # CPU test
    with tf.device('/CPU:0'):
        # Warm-up
        c = tf.matmul(a, b)
        
        # Actual test
        start_time = time.time()
        c = tf.matmul(a, b)
        _ = c.numpy()  # Force execution
        cpu_time = time.time() - start_time
        print(f"  CPU time: {cpu_time:.4f} seconds")
    
    # GPU test
    gpus = tf.config.list_physical_devices('GPU')
    if len(gpus) > 0:
        with tf.device('/GPU:0'):
            # Warm-up
            c = tf.matmul(a, b)
            
            # Actual test
            start_time = time.time()
            c = tf.matmul(a, b)
            _ = c.numpy()  # Force execution
            gpu_time = time.time() - start_time
            speedup = cpu_time / gpu_time if gpu_time > 0 else 0
            print(f"  GPU time: {gpu_time:.4f} seconds")
            print(f"  Speedup factor: {speedup:.2f}x")
            return speedup
    
    return 0

def run_memory_benchmark(size=10_000_000):
    """Test memory transfer between CPU and GPU."""
    gpus = tf.config.list_physical_devices('GPU')
    if len(gpus) == 0:
        return 0, 0
    
    print(f"Running memory transfer test (array size: {size})...")
    
    # Create data on CPU
    cpu_array = np.random.random(size).astype(np.float32)
    array_size_gb = cpu_array.nbytes / (1024 ** 3)
    
    # Measure CPU to GPU transfer time
    start_time = time.time()
    gpu_array = tf.constant(cpu_array, device='/GPU:0')
    _ = gpu_array.device  # Force completion
    cpu_to_gpu_time = time.time() - start_time
    
    # Calculate transfer rate
    transfer_rate_cpu_to_gpu = array_size_gb / cpu_to_gpu_time if cpu_to_gpu_time > 0 else 0
    
    # Measure GPU to CPU transfer time
    start_time = time.time()
    cpu_result = gpu_array.numpy()
    gpu_to_cpu_time = time.time() - start_time
    
    # Calculate transfer rate
    transfer_rate_gpu_to_cpu = array_size_gb / gpu_to_cpu_time if gpu_to_cpu_time > 0 else 0
    
    print(f"  CPU to GPU transfer rate: {transfer_rate_cpu_to_gpu:.2f} GB/s")
    print(f"  GPU to CPU transfer rate: {transfer_rate_gpu_to_cpu:.2f} GB/s")
    
    return transfer_rate_cpu_to_gpu, transfer_rate_gpu_to_cpu

def calculate_score(matrix_speedup, memory_rate):
    """Calculate a simple benchmark score."""
    if matrix_speedup <= 0:
        return 0
    
    # Base score calculation
    base_score = 1000
    matrix_component = base_score * matrix_speedup * 0.7
    memory_component = base_score * memory_rate * 5.0 * 0.3
    
    return int(matrix_component + memory_component)

def main():
    """Run the quick benchmark."""
    print("=" * 60)
    print("TensorFlow GPU Quick Benchmark")
    print("=" * 60)
    
    # Configure GPU
    gpus = configure_gpu()
    
    # Get system info
    system_info = get_system_info()
    print("\nSystem Information:")
    for key, value in system_info.items():
        print(f"  {key}: {value}")
    
    # Check for GPU
    if len(gpus) == 0:
        print("\n❌ No GPU detected. Benchmark will run on CPU only.")
        return
    
    print(f"\n✅ Found {len(gpus)} GPU(s):")
    for i, gpu in enumerate(gpus):
        print(f"  GPU {i}: {gpu.name}")
    
    # Run quick benchmarks
    print("\nRunning quick benchmarks...")
    matrix_speedup = run_matrix_multiplication_test()
    memory_rate_to_gpu, memory_rate_from_gpu = run_memory_benchmark()
    
    # Calculate score
    score = calculate_score(matrix_speedup, memory_rate_to_gpu)
    
    print("\n" + "=" * 60)
    print(f"Quick Benchmark Score: {score}")
    print("=" * 60)
    
    # Save results
    if not os.path.exists("benchmark_results"):
        os.makedirs("benchmark_results")
    
    results = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "system_info": system_info,
        "matrix_speedup": matrix_speedup,
        "memory_rate_to_gpu": memory_rate_to_gpu,
        "memory_rate_from_gpu": memory_rate_from_gpu,
        "score": score
    }
    
    filename = f"benchmark_results/quick_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {filename}")
    print("\nQuick benchmark complete!")

if __name__ == "__main__":
    main()