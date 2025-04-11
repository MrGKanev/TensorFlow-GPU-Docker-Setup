#!/usr/bin/env python3
"""
TensorFlow GPU Benchmark Suite

This script runs a comprehensive suite of performance tests to benchmark
TensorFlow performance with GPU acceleration. It's designed to measure:
- Basic GPU operations speed
- Matrix multiplication performance at various sizes
- CNN training and inference speed
- Memory transfer rates
- LSTM performance
- Multi-GPU scaling (if available)

Results are output in a format that makes it easy to compare across systems.
"""

import os
import time
import json
import platform
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
from tensorflow.keras import layers, models, datasets
from tensorflow.keras.optimizers.legacy import Adam

# Ensure TensorFlow logs only errors, not warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class TensorFlowBenchmark:
    def __init__(self):
        """Initialize the benchmark class and detect available hardware."""
        self.results = {
            "system_info": self.get_system_info(),
            "gpu_info": self.get_gpu_info(),
            "tests": {},
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Configure GPU memory growth to avoid OOM errors
        self.configure_gpu()
        
        # Create results directory if it doesn't exist
        if not os.path.exists("benchmark_results"):
            os.makedirs("benchmark_results")
    
    def configure_gpu(self):
        """Configure GPUs to use memory growth and prevent OOM errors."""
        self.gpus = tf.config.list_physical_devices('GPU')
        if self.gpus:
            try:
                for gpu in self.gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"✅ GPU memory growth enabled on {len(self.gpus)} GPUs")
            except RuntimeError as e:
                print(f"❌ Error setting GPU memory growth: {e}")
    
    def get_system_info(self):
        """Gather system information."""
        system_info = {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "tensorflow_version": tf.__version__,
            "is_built_with_cuda": tf.test.is_built_with_cuda(),
            "is_gpu_available": tf.test.is_gpu_available() if hasattr(tf.test, 'is_gpu_available') else len(tf.config.list_physical_devices('GPU')) > 0
        }
        return system_info
    
    def get_gpu_info(self):
        """Get information about available GPUs."""
        gpus = tf.config.list_physical_devices('GPU')
        gpu_info = {
            "gpu_count": len(gpus),
            "devices": []
        }
        
        if gpus:
            for i, gpu in enumerate(gpus):
                try:
                    # Get GPU details if available
                    details = tf.config.experimental.get_device_details(gpu) if hasattr(tf.config.experimental, 'get_device_details') else {"name": gpu.name}
                    gpu_info["devices"].append(details)
                except:
                    gpu_info["devices"].append({"name": f"GPU:{i}", "info": "Details not available"})
            
            # Try to get CUDA version
            try:
                gpu_info["cuda_version"] = tf.sysconfig.get_build_info()["cuda_version"]
            except:
                gpu_info["cuda_version"] = "Unknown"
        
        return gpu_info
    
    def run_all_benchmarks(self):
        """Run all benchmark tests."""
        print("=" * 50)
        print("TensorFlow GPU Benchmark Suite")
        print("=" * 50)
        
        # Print system information
        print("\nSystem Information:")
        for key, value in self.results["system_info"].items():
            print(f"  {key}: {value}")
        
        # Print GPU information
        print("\nGPU Information:")
        print(f"  GPU Count: {self.results['gpu_info']['gpu_count']}")
        if self.results["gpu_info"]["gpu_count"] > 0:
            for i, device in enumerate(self.results["gpu_info"]["devices"]):
                print(f"  GPU {i}: {device.get('name', 'Unknown')}")
            print(f"  CUDA Version: {self.results['gpu_info'].get('cuda_version', 'Unknown')}")
        
        # Run the benchmarks
        print("\nRunning benchmarks...")
        
        # Test 1: Matrix multiplication benchmark
        self.run_matrix_multiplication_benchmark()
        
        # Test 2: CNN inference benchmark
        self.run_cnn_inference_benchmark()
        
        # Test 3: CNN training benchmark 
        self.run_cnn_training_benchmark()
        
        # Test 4: Memory transfer benchmark
        self.run_memory_transfer_benchmark()
        
        # Test 5: LSTM benchmark
        self.run_lstm_benchmark()
        
        # Test 6: Multi-GPU benchmark (if available)
        if self.results["gpu_info"]["gpu_count"] > 1:
            self.run_multi_gpu_benchmark()
        
        # Generate the final report
        self.save_results()
        self.print_summary()
    
    def run_matrix_multiplication_benchmark(self):
        """Benchmark matrix multiplication performance at different sizes."""
        print("\nRunning Matrix Multiplication Benchmark...")
        
        # Matrix sizes to test
        sizes = [128, 512, 1024, 2048, 4096]
        cpu_times = []
        gpu_times = []
        
        results = {
            "sizes": sizes,
            "cpu_times": [],
            "gpu_times": [],
            "speedup_factors": []
        }
        
        for size in sizes:
            print(f"  Testing matrix size: {size}x{size}")
            
            # Create random matrices
            a = tf.random.normal([size, size])
            b = tf.random.normal([size, size])
            
            # CPU test
            with tf.device('/CPU:0'):
                # Warm-up
                c = tf.matmul(a, b)
                
                # Actual test
                start_time = time.time()
                for _ in range(3):  # Run multiple times for more stable results
                    c = tf.matmul(a, b)
                    # Force execution to complete
                    _ = c.numpy()
                cpu_time = (time.time() - start_time) / 3
                results["cpu_times"].append(cpu_time)
                print(f"    CPU time: {cpu_time:.4f} seconds")
            
            # GPU test (if available)
            if self.results["gpu_info"]["gpu_count"] > 0:
                with tf.device('/GPU:0'):
                    # Warm-up
                    c = tf.matmul(a, b)
                    
                    # Actual test
                    start_time = time.time()
                    for _ in range(5):  # Run multiple times for more stable results
                        c = tf.matmul(a, b)
                        # Force execution to complete
                        _ = c.numpy()
                    gpu_time = (time.time() - start_time) / 5
                    results["gpu_times"].append(gpu_time)
                    speedup = cpu_time / gpu_time if gpu_time > 0 else 0
                    results["speedup_factors"].append(speedup)
                    print(f"    GPU time: {gpu_time:.4f} seconds")
                    print(f"    Speedup factor: {speedup:.2f}x")
            else:
                results["gpu_times"].append(None)
                results["speedup_factors"].append(None)
        
        self.results["tests"]["matrix_multiplication"] = results
    
    def run_cnn_inference_benchmark(self):
        """Benchmark CNN inference performance."""
        print("\nRunning CNN Inference Benchmark...")
        
        # Create a simple CNN model
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(10)
        ])
        
        # Compile the model
        model.compile(optimizer='adam',
                     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                     metrics=['accuracy'])
        
        # Create synthetic test data
        test_images = tf.random.normal([1000, 32, 32, 3])
        
        results = {
            "batch_sizes": [1, 8, 16, 32, 64, 128],
            "cpu_times": [],
            "gpu_times": [],
            "speedup_factors": []
        }
        
        for batch_size in results["batch_sizes"]:
            print(f"  Testing batch size: {batch_size}")
            
            # CPU test
            with tf.device('/CPU:0'):
                # Warm-up
                _ = model.predict(test_images[:batch_size])
                
                # Actual test
                start_time = time.time()
                _ = model.predict(test_images[:batch_size])
                cpu_time = time.time() - start_time
                results["cpu_times"].append(cpu_time)
                print(f"    CPU time: {cpu_time:.4f} seconds")
            
            # GPU test (if available)
            if self.results["gpu_info"]["gpu_count"] > 0:
                with tf.device('/GPU:0'):
                    # Warm-up
                    _ = model.predict(test_images[:batch_size])
                    
                    # Actual test
                    start_time = time.time()
                    _ = model.predict(test_images[:batch_size])
                    gpu_time = time.time() - start_time
                    results["gpu_times"].append(gpu_time)
                    speedup = cpu_time / gpu_time if gpu_time > 0 else 0
                    results["speedup_factors"].append(speedup)
                    print(f"    GPU time: {gpu_time:.4f} seconds")
                    print(f"    Speedup factor: {speedup:.2f}x")
            else:
                results["gpu_times"].append(None)
                results["speedup_factors"].append(None)
        
        self.results["tests"]["cnn_inference"] = results
    
    def run_cnn_training_benchmark(self):
        """Benchmark CNN training performance."""
        print("\nRunning CNN Training Benchmark...")
        
        # Create a simple CNN model
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(10)
        ])
        
        # Compile the model with legacy Adam optimizer for TF 2.11 compatibility
        model.compile(optimizer=Adam(),
                     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                     metrics=['accuracy'])
        
        # Create synthetic training data
        train_images = tf.random.normal([1000, 32, 32, 3])
        train_labels = tf.random.uniform([1000], minval=0, maxval=10, dtype=tf.int32)
        
        results = {
            "batch_sizes": [16, 32, 64, 128],
            "cpu_times": [],
            "gpu_times": [],
            "speedup_factors": []
        }
        
        for batch_size in results["batch_sizes"]:
            print(f"  Testing batch size: {batch_size}")
            
            # CPU test
            with tf.device('/CPU:0'):
                # Create a fresh model to ensure fair comparison
                model_cpu = models.Sequential([
                    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
                    layers.MaxPooling2D((2, 2)),
                    layers.Conv2D(64, (3, 3), activation='relu'),
                    layers.MaxPooling2D((2, 2)),
                    layers.Conv2D(64, (3, 3), activation='relu'),
                    layers.Flatten(),
                    layers.Dense(64, activation='relu'),
                    layers.Dense(10)
                ])
                model_cpu.compile(optimizer=Adam(),
                                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                                metrics=['accuracy'])
                
                # Actual test - train for 3 epochs
                start_time = time.time()
                model_cpu.fit(train_images, train_labels, epochs=3, batch_size=batch_size, verbose=0)
                cpu_time = time.time() - start_time
                results["cpu_times"].append(cpu_time)
                print(f"    CPU time: {cpu_time:.4f} seconds")
            
            # GPU test (if available)
            if self.results["gpu_info"]["gpu_count"] > 0:
                with tf.device('/GPU:0'):
                    # Create a fresh model to ensure fair comparison
                    model_gpu = models.Sequential([
                        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
                        layers.MaxPooling2D((2, 2)),
                        layers.Conv2D(64, (3, 3), activation='relu'),
                        layers.MaxPooling2D((2, 2)),
                        layers.Conv2D(64, (3, 3), activation='relu'),
                        layers.Flatten(),
                        layers.Dense(64, activation='relu'),
                        layers.Dense(10)
                    ])
                    model_gpu.compile(optimizer=Adam(),
                                   loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                                   metrics=['accuracy'])
                    
                    # Actual test - train for 3 epochs
                    start_time = time.time()
                    model_gpu.fit(train_images, train_labels, epochs=3, batch_size=batch_size, verbose=0)
                    gpu_time = time.time() - start_time
                    results["gpu_times"].append(gpu_time)
                    speedup = cpu_time / gpu_time if gpu_time > 0 else 0
                    results["speedup_factors"].append(speedup)
                    print(f"    GPU time: {gpu_time:.4f} seconds")
                    print(f"    Speedup factor: {speedup:.2f}x")
            else:
                results["gpu_times"].append(None)
                results["speedup_factors"].append(None)
        
        self.results["tests"]["cnn_training"] = results
    
    def run_memory_transfer_benchmark(self):
        """Benchmark memory transfer speeds between CPU and GPU."""
        if self.results["gpu_info"]["gpu_count"] == 0:
            print("\nSkipping Memory Transfer Benchmark (No GPU available)")
            return
        
        print("\nRunning Memory Transfer Benchmark...")
        
        # Define array sizes to test
        sizes = [1_000_000, 10_000_000, 100_000_000, 500_000_000]
        results = {
            "sizes": sizes,
            "cpu_to_gpu_times": [],
            "gpu_to_cpu_times": [],
            "transfer_rates_cpu_to_gpu": [],  # in GB/s
            "transfer_rates_gpu_to_cpu": []   # in GB/s
        }
        
        for size in sizes:
            print(f"  Testing array size: {size} elements")
            
            # Create data on CPU
            cpu_array = np.random.random(size).astype(np.float32)
            array_size_gb = cpu_array.nbytes / (1024 ** 3)
            
            # Measure CPU to GPU transfer time
            start_time = time.time()
            gpu_array = tf.constant(cpu_array, device='/GPU:0')
            # Force completion
            _ = gpu_array.device
            cpu_to_gpu_time = time.time() - start_time
            
            # Calculate transfer rate
            transfer_rate_cpu_to_gpu = array_size_gb / cpu_to_gpu_time if cpu_to_gpu_time > 0 else 0
            
            # Measure GPU to CPU transfer time
            start_time = time.time()
            cpu_result = gpu_array.numpy()
            gpu_to_cpu_time = time.time() - start_time
            
            # Calculate transfer rate
            transfer_rate_gpu_to_cpu = array_size_gb / gpu_to_cpu_time if gpu_to_cpu_time > 0 else 0
            
            # Store results
            results["cpu_to_gpu_times"].append(cpu_to_gpu_time)
            results["gpu_to_cpu_times"].append(gpu_to_cpu_time)
            results["transfer_rates_cpu_to_gpu"].append(transfer_rate_cpu_to_gpu)
            results["transfer_rates_gpu_to_cpu"].append(transfer_rate_gpu_to_cpu)
            
            print(f"    CPU to GPU time: {cpu_to_gpu_time:.4f} seconds")
            print(f"    CPU to GPU transfer rate: {transfer_rate_cpu_to_gpu:.2f} GB/s")
            print(f"    GPU to CPU time: {gpu_to_cpu_time:.4f} seconds")
            print(f"    GPU to CPU transfer rate: {transfer_rate_gpu_to_cpu:.2f} GB/s")
        
        self.results["tests"]["memory_transfer"] = results
    
    def run_lstm_benchmark(self):
        """Benchmark LSTM network performance."""
        print("\nRunning LSTM Benchmark...")
        
        # Define parameters for the test
        sequence_length = 100
        feature_dim = 32
        batch_sizes = [16, 32, 64, 128]
        
        results = {
            "batch_sizes": batch_sizes,
            "cpu_times": [],
            "gpu_times": [],
            "speedup_factors": []
        }
        
        # Create synthetic data
        input_data = np.random.random((1000, sequence_length, feature_dim)).astype(np.float32)
        targets = np.random.random((1000, 1)).astype(np.float32)
        
        for batch_size in batch_sizes:
            print(f"  Testing batch size: {batch_size}")
            
            # CPU test
            with tf.device('/CPU:0'):
                # Create a fresh model
                model_cpu = models.Sequential([
                    layers.LSTM(128, return_sequences=True, input_shape=(sequence_length, feature_dim)),
                    layers.LSTM(64),
                    layers.Dense(1)
                ])
                model_cpu.compile(optimizer=Adam(), loss='mse')
                
                # Train for one epoch
                start_time = time.time()
                model_cpu.fit(input_data, targets, epochs=1, batch_size=batch_size, verbose=0)
                cpu_time = time.time() - start_time
                results["cpu_times"].append(cpu_time)
                print(f"    CPU time: {cpu_time:.4f} seconds")
            
            # GPU test (if available)
            if self.results["gpu_info"]["gpu_count"] > 0:
                with tf.device('/GPU:0'):
                    # Create a fresh model
                    model_gpu = models.Sequential([
                        layers.LSTM(128, return_sequences=True, input_shape=(sequence_length, feature_dim)),
                        layers.LSTM(64),
                        layers.Dense(1)
                    ])
                    model_gpu.compile(optimizer=Adam(), loss='mse')
                    
                    # Train for one epoch
                    start_time = time.time()
                    model_gpu.fit(input_data, targets, epochs=1, batch_size=batch_size, verbose=0)
                    gpu_time = time.time() - start_time
                    results["gpu_times"].append(gpu_time)
                    speedup = cpu_time / gpu_time if gpu_time > 0 else 0
                    results["speedup_factors"].append(speedup)
                    print(f"    GPU time: {gpu_time:.4f} seconds")
                    print(f"    Speedup factor: {speedup:.2f}x")
            else:
                results["gpu_times"].append(None)
                results["speedup_factors"].append(None)
        
        self.results["tests"]["lstm"] = results
    
    def run_multi_gpu_benchmark(self):
        """Benchmark performance scaling with multiple GPUs."""
        if self.results["gpu_info"]["gpu_count"] <= 1:
            return
        
        print("\nRunning Multi-GPU Benchmark...")
        
        # Define the model for testing
        def create_model():
            return models.Sequential([
                layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation='relu'),
                layers.Flatten(),
                layers.Dense(64, activation='relu'),
                layers.Dense(10)
            ])
        
        # Create synthetic data
        train_images = tf.random.normal([2000, 32, 32, 3])
        train_labels = tf.random.uniform([2000], minval=0, maxval=10, dtype=tf.int32)
        
        # Test with different numbers of GPUs
        gpu_counts = list(range(1, self.results["gpu_info"]["gpu_count"] + 1))
        results = {
            "gpu_counts": gpu_counts,
            "training_times": [],
            "speedup_factors": []
        }
        
        # Baseline: Single GPU performance
        with tf.device('/GPU:0'):
            model = create_model()
            model.compile(optimizer=Adam(),
                         loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                         metrics=['accuracy'])
            
            start_time = time.time()
            model.fit(train_images, train_labels, epochs=3, batch_size=64, verbose=0)
            single_gpu_time = time.time() - start_time
            results["training_times"].append(single_gpu_time)
            results["speedup_factors"].append(1.0)  # Reference speedup
            
            print(f"  Single GPU time: {single_gpu_time:.4f} seconds")
        
        # Skip if TensorFlow version doesn't support Distribution Strategy
        if not hasattr(tf.distribute, 'MirroredStrategy'):
            print("  Multi-GPU testing skipped (TensorFlow version doesn't support Distribution Strategy)")
            self.results["tests"]["multi_gpu"] = results
            return
        
        # Test with multiple GPUs
        for num_gpus in range(2, self.results["gpu_info"]["gpu_count"] + 1):
            print(f"  Testing with {num_gpus} GPUs")
            
            try:
                # Create a MirroredStrategy
                strategy = tf.distribute.MirroredStrategy(devices=[f'/GPU:{i}' for i in range(num_gpus)])
                
                with strategy.scope():
                    model = create_model()
                    model.compile(optimizer=Adam(),
                                 loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                                 metrics=['accuracy'])
                
                start_time = time.time()
                model.fit(train_images, train_labels, epochs=3, batch_size=64*num_gpus, verbose=0)
                multi_gpu_time = time.time() - start_time
                
                speedup = single_gpu_time / multi_gpu_time if multi_gpu_time > 0 else 0
                results["training_times"].append(multi_gpu_time)
                results["speedup_factors"].append(speedup)
                
                print(f"    {num_gpus} GPU time: {multi_gpu_time:.4f} seconds")
                print(f"    Speedup factor vs 1 GPU: {speedup:.2f}x")
                print(f"    Efficiency: {(speedup/num_gpus)*100:.1f}%")
                
            except Exception as e:
                print(f"    Error testing {num_gpus} GPUs: {e}")
                results["training_times"].append(None)
                results["speedup_factors"].append(None)
        
        self.results["tests"]["multi_gpu"] = results
    
    def save_results(self):
        """Save benchmark results to a JSON file."""
        filename = f"benchmark_results/tf_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\nResults saved to {filename}")
        
        # Also save simple results summary
        summary_filename = f"benchmark_results/tf_benchmark_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(summary_filename, 'w') as f:
            f.write("TensorFlow GPU Benchmark Summary\n")
            f.write("=" * 40 + "\n\n")
            
            # System info
            f.write("System Information:\n")
            for key, value in self.results["system_info"].items():
                f.write(f"  {key}: {value}\n")
            
            # GPU info
            f.write("\nGPU Information:\n")
            f.write(f"  GPU Count: {self.results['gpu_info']['gpu_count']}\n")
            if self.results["gpu_info"]["gpu_count"] > 0:
                for i, device in enumerate(self.results["gpu_info"]["devices"]):
                    f.write(f"  GPU {i}: {device.get('name', 'Unknown')}\n")
            
            # Test summaries
            if "matrix_multiplication" in self.results["tests"]:
                f.write("\nMatrix Multiplication:\n")
                data = self.results["tests"]["matrix_multiplication"]
                for i, size in enumerate(data["sizes"]):
                    speedup = data["speedup_factors"][i]
                    if speedup is not None:
                        f.write(f"  {size}x{size}: {speedup:.2f}x speedup\n")
            
            if "cnn_inference" in self.results["tests"]:
                f.write("\nCNN Inference:\n")
                data = self.results["tests"]["cnn_inference"]
                for i, batch_size in enumerate(data["batch_sizes"]):
                    speedup = data["speedup_factors"][i]
                    if speedup is not None:
                        f.write(f"  Batch size {batch_size}: {speedup:.2f}x speedup\n")
            
            if "cnn_training" in self.results["tests"]:
                f.write("\nCNN Training:\n")
                data = self.results["tests"]["cnn_training"]
                for i, batch_size in enumerate(data["batch_sizes"]):
                    speedup = data["speedup_factors"][i]
                    if speedup is not None:
                        f.write(f"  Batch size {batch_size}: {speedup:.2f}x speedup\n")
            
            if "memory_transfer" in self.results["tests"]:
                f.write("\nMemory Transfer Rates:\n")
                data = self.results["tests"]["memory_transfer"]
                for i, size in enumerate(data["sizes"]):
                    cpu_to_gpu = data["transfer_rates_cpu_to_gpu"][i]
                    gpu_to_cpu = data["transfer_rates_gpu_to_cpu"][i]
                    f.write(f"  Array size {size}: CPU→GPU {cpu_to_gpu:.2f} GB/s, GPU→CPU {gpu_to_cpu:.2f} GB/s\n")
            
            if "lstm" in self.results["tests"]:
                f.write("\nLSTM Performance:\n")
                data = self.results["tests"]["lstm"]
                for i, batch_size in enumerate(data["batch_sizes"]):
                    speedup = data["speedup_factors"][i]
                    if speedup is not None:
                        f.write(f"  Batch size {batch_size}: {speedup:.2f}x speedup\n")
            
            if "multi_gpu" in self.results["tests"]:
                f.write("\nMulti-GPU Scaling:\n")
                data = self.results["tests"]["multi_gpu"]
                for i, num_gpus in enumerate(data["gpu_counts"]):
                    speedup = data["speedup_factors"][i]
                    if speedup is not None:
                        f.write(f"  {num_gpus} GPUs: {speedup:.2f}x speedup\n")
            
            f.write("\n" + "=" * 40 + "\n")
            f.write(f"Benchmark completed: {self.results['timestamp']}\n")
        
        print(f"Summary saved to {summary_filename}")
    
    def print_summary(self):
        """Print a summary of the benchmark results."""
        print("\n" + "=" * 50)
        print("TensorFlow GPU Benchmark Summary")
        print("=" * 50)
        
        # Matrix multiplication
        if "matrix_multiplication" in self.results["tests"]:
            print("\nMatrix Multiplication Speedup Factors:")
            data = self.results["tests"]["matrix_multiplication"]
            for i, size in enumerate(data["sizes"]):
                speedup = data["speedup_factors"][i]
                if speedup is not None:
                    print(f"  {size}x{size}: {speedup:.2f}x speedup")
        
        # CNN inference
        if "cnn_inference" in self.results["tests"]:
            print("\nCNN Inference Speedup Factors:")
            data = self.results["tests"]["cnn_inference"]
            for i, batch_size in enumerate(data["batch_sizes"]):
                speedup = data["speedup_factors"][i]
                if speedup is not None:
                    print(f"  Batch size {batch_size}: {speedup:.2f}x speedup")
        
        # CNN training
        if "cnn_training" in self.results["tests"]:
            print("\nCNN Training Speedup Factors:")
            data = self.results["tests"]["cnn_training"]
            for i, batch_size in enumerate(data["batch_sizes"]):
                speedup = data["speedup_factors"][i]
                if speedup is not None:
                    print(f"  Batch size {batch_size}: {speedup:.2f}x speedup")
        
        # Memory transfer
        if "memory_transfer" in self.results["tests"]:
            print("\nMemory Transfer Rates:")
            data = self.results["tests"]["memory_transfer"]
            for i, size in enumerate(data["sizes"]):
                cpu_to_gpu = data["transfer_rates_cpu_to_gpu"][i]
                gpu_to_cpu = data["transfer_rates_gpu_to_cpu"][i]
                print(f"  Array size {size}: CPU→GPU {cpu_to_gpu:.2f} GB/s, GPU→CPU {gpu_to_cpu:.2f} GB/s")
        
        # LSTM
        if "lstm" in self.results["tests"]:
            print("\nLSTM Speedup Factors:")
            data = self.results["tests"]["lstm"]
            for i, batch_size in enumerate(data["batch_sizes"]):
                speedup = data["speedup_factors"][i]
                if speedup is not None:
                    print(f"  Batch size {batch_size}: {speedup:.2f}x speedup")
        
        # Multi-GPU
        if "multi_gpu" in self.results["tests"]:
            print("\nMulti-GPU Scaling:")
            data = self.results["tests"]["multi_gpu"]
            for i, num_gpus in enumerate(data["gpu_counts"]):
                speedup = data["speedup_factors"][i]
                if speedup is not None:
                    efficiency = (speedup/num_gpus)*100
                    print(f"  {num_gpus} GPUs: {speedup:.2f}x speedup (Efficiency: {efficiency:.1f}%)")
        
        print("\n" + "=" * 50)
        print(f"Benchmark completed: {self.results['timestamp']}")
        print("=" * 50)

def generate_charts(self):
    """Generate charts from benchmark results."""
    # Skip if matplotlib is not available
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib not installed, skipping chart generation")
        return
    
    plt.style.use('ggplot')
    
    # Create a directory for charts
    if not os.path.exists("benchmark_results/charts"):
        os.makedirs("benchmark_results/charts")
    
    # Matrix multiplication chart
    if "matrix_multiplication" in self.results["tests"]:
        data = self.results["tests"]["matrix_multiplication"]
        
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(data["sizes"])), data["speedup_factors"])
        plt.xlabel('Matrix Size')
        plt.ylabel('GPU Speedup (x times)')
        plt.title('Matrix Multiplication: GPU vs CPU Speedup')
        plt.xticks(range(len(data["sizes"])), [f"{size}x{size}" for size in data["sizes"]])
        
        for i, speedup in enumerate(data["speedup_factors"]):
            if speedup is not None:
                plt.text(i, speedup + 0.5, f"{speedup:.1f}x", ha='center')
        
        plt.savefig("benchmark_results/charts/matrix_mult_speedup.png")
    
    # CNN charts (inference and training)
    for test_name in ["cnn_inference", "cnn_training"]:
        if test_name in self.results["tests"]:
            data = self.results["tests"][test_name]
            
            plt.figure(figsize=(10, 6))
            plt.bar(range(len(data["batch_sizes"])), data["speedup_factors"])
            plt.xlabel('Batch Size')
            plt.ylabel('GPU Speedup (x times)')
            plt.title(f'{test_name.replace("_", " ").title()}: GPU vs CPU Speedup')
            plt.xticks(range(len(data["batch_sizes"])), data["batch_sizes"])
            
            for i, speedup in enumerate(data["speedup_factors"]):
                if speedup is not None:
                    plt.text(i, speedup + 0.5, f"{speedup:.1f}x", ha='center')
            
            plt.savefig(f"benchmark_results/charts/{test_name}_speedup.png")
    
    # Memory transfer rates
    if "memory_transfer" in self.results["tests"]:
        data = self.results["tests"]["memory_transfer"]
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(data["sizes"])), data["transfer_rates_cpu_to_gpu"], 'o-', label='CPU to GPU')
        plt.plot(range(len(data["sizes"])), data["transfer_rates_gpu_to_cpu"], 's-', label='GPU to CPU')
        plt.xlabel('Array Size')
        plt.ylabel('Transfer Rate (GB/s)')
        plt.title('Memory Transfer Rates')
        plt.xticks(range(len(data["sizes"])), [f"{size/1_000_000:.1f}M" for size in data["sizes"]])
        plt.legend()
        
        plt.savefig("benchmark_results/charts/memory_transfer.png")
    
    # LSTM speedup
    if "lstm" in self.results["tests"]:
        data = self.results["tests"]["lstm"]
        
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(data["batch_sizes"])), data["speedup_factors"])
        plt.xlabel('Batch Size')
        plt.ylabel('GPU Speedup (x times)')
        plt.title('LSTM: GPU vs CPU Speedup')
        plt.xticks(range(len(data["batch_sizes"])), data["batch_sizes"])
        
        for i, speedup in enumerate(data["speedup_factors"]):
            if speedup is not None:
                plt.text(i, speedup + 0.5, f"{speedup:.1f}x", ha='center')
        
        plt.savefig("benchmark_results/charts/lstm_speedup.png")
    
    # Multi-GPU scaling
    if "multi_gpu" in self.results["tests"]:
        data = self.results["tests"]["multi_gpu"]
        
        plt.figure(figsize=(10, 6))
        plt.plot(data["gpu_counts"], data["speedup_factors"], 'o-')
        plt.plot(data["gpu_counts"], data["gpu_counts"], '--', label='Linear Scaling (Ideal)')
        plt.xlabel('Number of GPUs')
        plt.ylabel('Speedup vs Single GPU')
        plt.title('Multi-GPU Scaling')
        plt.xticks(data["gpu_counts"])
        plt.legend()
        
        plt.savefig("benchmark_results/charts/multi_gpu_scaling.png")
    
    print("Charts generated in benchmark_results/charts/ directory")

def generate_score(self):
    """Generate a single benchmark score, similar to Geekbench."""
    if self.results["gpu_info"]["gpu_count"] == 0:
        print("Cannot generate score (No GPU available)")
        return 0
    
    # Base score (scaled to approximately match Geekbench GPU Compute score range)
    base_score = 1000
    score_components = []
    
    # Matrix multiplication component - focused on 2048x2048 matrix
    if "matrix_multiplication" in self.results["tests"]:
        data = self.results["tests"]["matrix_multiplication"]
        # Find the closest size to 2048
        idx = min(range(len(data["sizes"])), key=lambda i: abs(data["sizes"][i] - 2048))
        if data["gpu_times"][idx] is not None and data["gpu_times"][idx] > 0:
            # Lower time is better
            mm_score = base_score * (1.0 / data["gpu_times"][idx]) * 0.1
            score_components.append(mm_score)
    
    # CNN training component
    if "cnn_training" in self.results["tests"]:
        data = self.results["tests"]["cnn_training"]
        # Use the largest batch size test
        idx = -1  # Last index (largest batch size)
        if data["gpu_times"][idx] is not None and data["gpu_times"][idx] > 0:
            # Lower time is better
            cnn_score = base_score * (1.0 / data["gpu_times"][idx]) * 0.25
            score_components.append(cnn_score)
    
    # Memory transfer component
    if "memory_transfer" in self.results["tests"]:
        data = self.results["tests"]["memory_transfer"]
        # Use the largest array size test
        idx = -1  # Last index (largest array size)
        # Higher transfer rate is better
        transfer_score = base_score * data["transfer_rates_cpu_to_gpu"][idx] * 0.05
        score_components.append(transfer_score)
    
    # LSTM component
    if "lstm" in self.results["tests"]:
        data = self.results["tests"]["lstm"]
        # Use the largest batch size test
        idx = -1  # Last index (largest batch size)
        if data["gpu_times"][idx] is not None and data["gpu_times"][idx] > 0:
            # Lower time is better
            lstm_score = base_score * (1.0 / data["gpu_times"][idx]) * 0.3
            score_components.append(lstm_score)
    
    # If we have multi-GPU results, add a scaling bonus
    if "multi_gpu" in self.results["tests"] and len(self.results["gpu_info"]["devices"]) > 1:
        data = self.results["tests"]["multi_gpu"]
        # Get the speedup from using all GPUs
        max_gpus_idx = -1  # Last index (all GPUs)
        if data["speedup_factors"][max_gpus_idx] is not None:
            # Apply a scaling bonus based on multi-GPU efficiency
            num_gpus = data["gpu_counts"][max_gpus_idx]
            speedup = data["speedup_factors"][max_gpus_idx]
            efficiency = speedup / num_gpus
            multi_gpu_score = base_score * speedup * 0.3 * efficiency
            score_components.append(multi_gpu_score)
    
    # Calculate final score
    if score_components:
        final_score = sum(score_components)
        
        # Store the score in results
        self.results["benchmark_score"] = int(final_score)
        
        print("\n" + "=" * 50)
        print(f"TensorFlow GPU Benchmark Score: {int(final_score)}")
        print("=" * 50)
        
        return int(final_score)
    else:
        print("Could not calculate benchmark score (insufficient data)")
        return 0


if __name__ == "__main__":
    benchmark = TensorFlowBenchmark()
    benchmark.run_all_benchmarks()
    benchmark.generate_score()
    
    # Generate charts if matplotlib is available
    try:
        benchmark.generate_charts()
    except Exception as e:
        print(f"Warning: Could not generate charts: {e}")
    
    print("\nBenchmark complete! Results saved to benchmark_results/ directory.")