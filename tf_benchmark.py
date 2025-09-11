#!/usr/bin/env python3
"""
TensorFlow GPU Benchmark

Enhanced benchmark script with support for:
- Mixed precision training
- XLA compilation
- Multi-GPU training
- Memory profiling
- Modern TensorFlow optimizations
"""

import time
import json
import psutil
import argparse
import numpy as np
import tensorflow as tf
from datetime import datetime
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.progress import Progress
from rich import print as rprint

console = Console()

class ModernBenchmark:
    def __init__(self, use_mixed_precision=True, use_xla=True):
        self.use_mixed_precision = use_mixed_precision
        self.use_xla = use_xla
        self.results = {}
        self.setup_tf_optimizations()
    
    def setup_tf_optimizations(self):
        """Configure TensorFlow for optimal performance."""
        # GPU memory growth
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        
        # Mixed precision
        if self.use_mixed_precision and gpus:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            console.print("✅ Mixed precision enabled", style="green")
        
        # XLA compilation
        if self.use_xla:
            tf.config.optimizer.set_jit(True)
            console.print("✅ XLA compilation enabled", style="green")
        
        # Thread optimization
        tf.config.threading.set_inter_op_parallelism_threads(0)
        tf.config.threading.set_intra_op_parallelism_threads(0)
    
    def print_system_info(self):
        """Print comprehensive system information."""
        table = Table(title="System Information")
        table.add_column("Component", style="cyan")
        table.add_column("Details", style="white")
        
        # TensorFlow info
        table.add_row("TensorFlow Version", tf.__version__)
        table.add_row("CUDA Available", str(tf.test.is_built_with_cuda()))
        
        # GPU info
        gpus = tf.config.list_physical_devices('GPU')
        table.add_row("GPUs Detected", str(len(gpus)))
        
        if gpus:
            for i, gpu in enumerate(gpus):
                try:
                    details = tf.config.experimental.get_device_details(gpu)
                    compute_cap = details.get('compute_capability', 'Unknown')
                    table.add_row(f"GPU {i}", f"{gpu.name} (Compute: {compute_cap})")
                except:
                    table.add_row(f"GPU {i}", gpu.name)
        
        # System info
        table.add_row("CPU Cores", str(psutil.cpu_count()))
        table.add_row("RAM", f"{psutil.virtual_memory().total / (1024**3):.1f} GB")
        
        console.print(table)
    
    @tf.function(experimental_relax_shapes=True)
    def compiled_matmul(self, a, b):
        """XLA-compiled matrix multiplication."""
        return tf.matmul(a, b)
    
    def benchmark_matrix_operations(self, size=5000, iterations=10):
        """Benchmark matrix operations with modern optimizations."""
        console.print(f"\n🔄 Matrix Operations Benchmark ({size}x{size})", style="bold blue")
        
        # Create test matrices
        with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
            a = tf.random.normal((size, size), dtype=tf.float32)
            b = tf.random.normal((size, size), dtype=tf.float32)
            
            if self.use_mixed_precision:
                a = tf.cast(a, tf.float16)
                b = tf.cast(b, tf.float16)
        
        # Warmup
        for _ in range(3):
            if self.use_xla:
                _ = self.compiled_matmul(a, b)
            else:
                _ = tf.matmul(a, b)
        
        # Benchmark
        start_time = time.time()
        with Progress() as progress:
            task = progress.add_task("Running benchmark...", total=iterations)
            for _ in range(iterations):
                if self.use_xla:
                    result = self.compiled_matmul(a, b)
                else:
                    result = tf.matmul(a, b)
                progress.advance(task)
        
        # Force execution and measure
        _ = result.numpy()
        elapsed = time.time() - start_time
        
        # Calculate metrics
        flops = 2 * size * size * size * iterations
        gflops = flops / elapsed / 1e9
        avg_time = elapsed / iterations
        
        self.results['matrix_ops'] = {
            'gflops': gflops,
            'avg_time_ms': avg_time * 1000,
            'total_time_s': elapsed
        }
        
        console.print(f"📊 Performance: {gflops:.2f} GFLOPS", style="green")
        console.print(f"⏱️  Average time: {avg_time*1000:.2f} ms", style="yellow")
        
        return gflops
    
    def benchmark_cnn_training(self, batch_size=64, epochs=5):
        """Benchmark CNN training with modern optimizations."""
        console.print(f"\n🧠 CNN Training Benchmark", style="bold blue")
        
        # Create a modern CNN model
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(224, 224, 3)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(64, 3, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(128, 3, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1000, activation='softmax')
        ])
        
        # Modern optimizer with learning rate scheduling
        initial_lr = 0.001
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_lr, decay_steps=100, decay_rate=0.9
        )
        
        optimizer = tf.keras.optimizers.AdamW(learning_rate=lr_schedule)
        
        # Mixed precision loss scaling
        if self.use_mixed_precision:
            optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
        
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Generate synthetic data
        x_train = tf.random.normal((batch_size * 10, 224, 224, 3))
        y_train = tf.random.uniform((batch_size * 10,), maxval=1000, dtype=tf.int32)
        
        # Benchmark training
        start_time = time.time()
        history = model.fit(
            x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=0
        )
        elapsed = time.time() - start_time
        
        samples_per_sec = (len(x_train) * epochs) / elapsed
        
        self.results['cnn_training'] = {
            'samples_per_sec': samples_per_sec,
            'time_per_epoch_s': elapsed / epochs,
            'final_accuracy': history.history['accuracy'][-1]
        }
        
        console.print(f"📊 Throughput: {samples_per_sec:.2f} samples/sec", style="green")
        console.print(f"🎯 Final accuracy: {history.history['accuracy'][-1]:.4f}", style="yellow")
        
        return samples_per_sec
    
    def benchmark_transformer_attention(self, seq_length=512, batch_size=32):
        """Benchmark transformer attention mechanism."""
        console.print(f"\n🔮 Transformer Attention Benchmark", style="bold blue")
        
        # Multi-head attention layer
        attention = tf.keras.layers.MultiHeadAttention(
            num_heads=8, key_dim=64, dropout=0.1
        )
        
        # Generate input data
        x = tf.random.normal((batch_size, seq_length, 512))
        
        # Warmup
        for _ in range(3):
            _ = attention(x, x)
        
        # Benchmark
        iterations = 50
        start_time = time.time()
        for _ in range(iterations):
            output = attention(x, x)
        _ = output.numpy()
        elapsed = time.time() - start_time
        
        tokens_per_sec = (batch_size * seq_length * iterations) / elapsed
        
        self.results['transformer_attention'] = {
            'tokens_per_sec': tokens_per_sec,
            'avg_time_ms': (elapsed / iterations) * 1000
        }
        
        console.print(f"📊 Throughput: {tokens_per_sec:.2f} tokens/sec", style="green")
        
        return tokens_per_sec
    
    def memory_profile(self):
        """Profile GPU memory usage."""
        console.print(f"\n💾 Memory Profile", style="bold blue")
        
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            console.print("No GPU available for memory profiling", style="red")
            return
        
        # Get memory info
        for i, gpu in enumerate(gpus):
            try:
                memory_info = tf.config.experimental.get_memory_info(gpu.name)
                current_mb = memory_info['current'] / (1024 * 1024)
                peak_mb = memory_info['peak'] / (1024 * 1024)
                
                table = Table(title=f"GPU {i} Memory Usage")
                table.add_column("Metric", style="cyan")
                table.add_column("Value", style="white")
                table.add_row("Current Usage", f"{current_mb:.1f} MB")
                table.add_row("Peak Usage", f"{peak_mb:.1f} MB")
                
                console.print(table)
                
                self.results[f'gpu_{i}_memory'] = {
                    'current_mb': current_mb,
                    'peak_mb': peak_mb
                }
            except Exception as e:
                console.print(f"Could not get memory info for GPU {i}: {e}", style="red")
    
    def save_results(self, filename=None):
        """Save benchmark results to JSON."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_results_{timestamp}.json"
        
        # Add metadata
        self.results['metadata'] = {
            'timestamp': datetime.now().isoformat(),
            'tensorflow_version': tf.__version__,
            'mixed_precision': self.use_mixed_precision,
            'xla_enabled': self.use_xla,
            'gpu_count': len(tf.config.list_physical_devices('GPU'))
        }
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        console.print(f"📄 Results saved to {filename}", style="green")
    
    def run_all_benchmarks(self):
        """Run comprehensive benchmark suite."""
        console.print("🚀 Starting Modern TensorFlow Benchmark", style="bold green")
        
        self.print_system_info()
        
        # Run benchmarks
        self.benchmark_matrix_operations()
        self.benchmark_cnn_training()
        self.benchmark_transformer_attention()
        self.memory_profile()
        
        # Save results
        self.save_results()
        
        console.print("\n✅ Benchmark completed successfully!", style="bold green")

def main():
    parser = argparse.ArgumentParser(description="Modern TensorFlow GPU Benchmark")
    parser.add_argument("--no-mixed-precision", action="store_true", 
                       help="Disable mixed precision training")
    parser.add_argument("--no-xla", action="store_true", 
                       help="Disable XLA compilation")
    parser.add_argument("--matrix-only", action="store_true", 
                       help="Run only matrix operations benchmark")
    parser.add_argument("--cnn-only", action="store_true", 
                       help="Run only CNN training benchmark")
    parser.add_argument("--attention-only", action="store_true", 
                       help="Run only transformer attention benchmark")
    
    args = parser.parse_args()
    
    # Initialize benchmark
    benchmark = ModernBenchmark(
        use_mixed_precision=not args.no_mixed_precision,
        use_xla=not args.no_xla
    )
    
    # Run specific benchmarks or all
    if args.matrix_only:
        benchmark.print_system_info()
        benchmark.benchmark_matrix_operations()
    elif args.cnn_only:
        benchmark.print_system_info()
        benchmark.benchmark_cnn_training()
    elif args.attention_only:
        benchmark.print_system_info()
        benchmark.benchmark_transformer_attention()
    else:
        benchmark.run_all_benchmarks()

if __name__ == "__main__":
    main()