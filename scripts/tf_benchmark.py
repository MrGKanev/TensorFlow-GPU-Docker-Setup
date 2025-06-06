#!/usr/bin/env python3
"""
TensorFlow Simple GPU Benchmark

A lightweight benchmark script for TensorFlow with GPU support.
No external dependencies beyond what's already in your container.
"""

import time
import numpy as np
import tensorflow as tf
import argparse
import os
import sys

def print_header(title):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)

def check_gpu():
    """Verify GPU availability and print information."""
    print_header("GPU Configuration")
    
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Python version: {sys.version.split()[0]}")
    print(f"CUDA available: {tf.test.is_built_with_cuda()}")
    
    # Check GPU devices
    gpus = tf.config.list_physical_devices('GPU')
    print(f"GPUs detected: {len(gpus)}")
    
    if gpus:
        print("\n✅ GPU IS AVAILABLE!")
        # Configure memory growth to avoid memory allocation errors
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
                print(f"Memory growth enabled for GPU {gpu}")
            except:
                print(f"Failed to set memory growth for GPU {gpu}")
        return True
    else:
        print("\n❌ NO GPU FOUND!")
        print("Make sure you started the container with --gpus all")
        return False

def benchmark_matrix_mult(size=5000, iters=10):
    """Benchmark matrix multiplication performance."""
    print_header(f"Matrix Multiplication ({size}x{size})")
    
    # Create random matrices
    print("Creating matrices...")
    a = tf.random.normal((size, size))
    b = tf.random.normal((size, size))
    
    # Warmup
    print("Warmup...")
    for _ in range(3):
        c = tf.matmul(a, b)
    
    # Benchmark
    print(f"Running benchmark ({iters} iterations)...")
    start_time = time.time()
    for _ in range(iters):
        c = tf.matmul(a, b)
    # Force execution before timing
    _ = c.numpy()
    end_time = time.time()
    
    # Calculate performance
    elapsed = end_time - start_time
    avg_time = elapsed / iters
    
    # Each matrix multiply is roughly 2*N^3 operations (N^2 multiplications and N^2 additions for N rows/columns)
    flops = 2 * size * size * size * iters
    gflops = flops / elapsed / 1e9
    
    print(f"\nResults:")
    print(f"  Total time: {elapsed:.3f} seconds")
    print(f"  Average per iteration: {avg_time*1000:.2f} ms")
    print(f"  Performance: {gflops:.2f} GFLOPS")
    
    return elapsed, gflops

def benchmark_dense_layer(batch_size=1024, input_size=1024, output_size=1024, iters=100):
    """Benchmark dense layer forward and backward pass."""
    print_header(f"Dense Layer ({input_size} → {output_size}, batch={batch_size})")
    
    # Create model with a single dense layer
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(output_size, input_shape=(input_size,), activation='relu')
    ])
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(),
        loss='mse'
    )
    
    # Create random input data
    x = tf.random.normal((batch_size, input_size))
    y = tf.random.normal((batch_size, output_size))
    
    # Warmup
    print("Warmup...")
    model.fit(x, y, epochs=2, batch_size=batch_size, verbose=0)
    
    # Benchmark
    print(f"Running benchmark ({iters} iterations)...")
    start_time = time.time()
    model.fit(x, y, epochs=iters, batch_size=batch_size, verbose=0)
    end_time = time.time()
    
    # Calculate performance
    elapsed = end_time - start_time
    avg_time = elapsed / iters
    
    # Each iteration processes batch_size samples
    samples_per_sec = batch_size * iters / elapsed
    
    print(f"\nResults:")
    print(f"  Total time: {elapsed:.3f} seconds")
    print(f"  Average per iteration: {avg_time*1000:.2f} ms")
    print(f"  Samples per second: {samples_per_sec:.2f}")
    
    return elapsed, samples_per_sec

def benchmark_conv_layer(batch_size=32, size=224, channels=3, filters=64, iters=10):
    """Benchmark convolutional layer performance."""
    print_header(f"Conv2D Layer ({size}x{size}x{channels} → {filters} filters)")
    
    # Create model with a single conv layer
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(
            filters=filters, 
            kernel_size=(3, 3), 
            activation='relu',
            padding='same',
            input_shape=(size, size, channels)
        )
    ])
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(),
        loss='mse'
    )
    
    # Create random input data
    x = tf.random.normal((batch_size, size, size, channels))
    y = tf.random.normal((batch_size, size, size, filters))
    
    # Warmup
    print("Warmup...")
    for _ in range(2):
        preds = model(x, training=False)
    
    # Benchmark inference
    print(f"Running inference benchmark ({iters} iterations)...")
    start_time = time.time()
    for _ in range(iters):
        preds = model(x, training=False)
    _ = preds.numpy()  # Force execution
    infer_time = time.time() - start_time
    
    # Benchmark training
    print(f"Running training benchmark ({iters} iterations)...")
    start_time = time.time()
    model.fit(x, y, epochs=iters, batch_size=batch_size, verbose=0)
    train_time = time.time() - start_time
    
    # Calculate performance
    infer_avg = infer_time / iters
    train_avg = train_time / iters
    
    infer_throughput = batch_size * iters / infer_time
    train_throughput = batch_size * iters / train_time
    
    print(f"\nInference Results:")
    print(f"  Total time: {infer_time:.3f} seconds")
    print(f"  Average per batch: {infer_avg*1000:.2f} ms")
    print(f"  Images per second: {infer_throughput:.2f}")
    
    print(f"\nTraining Results:")
    print(f"  Total time: {train_time:.3f} seconds")
    print(f"  Average per batch: {train_avg*1000:.2f} ms")
    print(f"  Images per second: {train_throughput:.2f}")
    
    return infer_time, train_time, infer_throughput, train_throughput

def run_all_benchmarks(device_name="GPU", use_small_sizes=False):
    """Run all benchmarks and display a summary."""
    results = {}
    
    if use_small_sizes:
        # Smaller sizes for faster runs
        matrix_size = 2000
        dense_input = 512
        dense_output = 512
        dense_batch = 128
        conv_size = 112  # 112x112 images
    else:
        # Standard sizes
        matrix_size = 5000
        dense_input = 1024
        dense_output = 1024
        dense_batch = 1024
        conv_size = 224  # 224x224 images
    
    # Matrix multiply benchmark
    _, matrix_gflops = benchmark_matrix_mult(matrix_size, iters=10)
    results["Matrix"] = matrix_gflops
    
    # Dense layer benchmark
    _, dense_sps = benchmark_dense_layer(
        batch_size=dense_batch, 
        input_size=dense_input, 
        output_size=dense_output, 
        iters=20
    )
    results["Dense"] = dense_sps
    
    # Conv layer benchmark
    _, _, _, conv_sps = benchmark_conv_layer(
        batch_size=32, 
        size=conv_size, 
        iters=10
    )
    results["Conv"] = conv_sps
    
    # Print summary
    print_header(f"Benchmark Summary ({device_name})")
    print(f"  Matrix Multiplication: {results['Matrix']:.2f} GFLOPS")
    print(f"  Dense Layer Throughput: {results['Dense']:.2f} samples/sec")
    print(f"  Conv2D Layer Throughput: {results['Conv']:.2f} images/sec")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Simple TensorFlow GPU Benchmark")
    parser.add_argument("--matrix", action="store_true", help="Run matrix multiplication benchmark")
    parser.add_argument("--dense", action="store_true", help="Run dense layer benchmark")
    parser.add_argument("--conv", action="store_true", help="Run convolution benchmark")
    parser.add_argument("--all", action="store_true", help="Run all benchmarks")
    parser.add_argument("--small", action="store_true", help="Use smaller sizes for quicker runs")
    args = parser.parse_args()
    
    # If no specific benchmark is requested, run all
    run_all = args.all or not (args.matrix or args.dense or args.conv)
    
    # Check for GPU
    has_gpu = check_gpu()
    device = "GPU" if has_gpu else "CPU"
    
    if args.matrix:
        benchmark_matrix_mult(2000 if args.small else 5000)
    
    if args.dense:
        batch = 128 if args.small else 1024
        size = 512 if args.small else 1024
        benchmark_dense_layer(batch_size=batch, input_size=size, output_size=size)
    
    if args.conv:
        size = 112 if args.small else 224
        benchmark_conv_layer(batch_size=32, size=size)
    
    if run_all:
        run_all_benchmarks(device, args.small)

if __name__ == "__main__":
    main()