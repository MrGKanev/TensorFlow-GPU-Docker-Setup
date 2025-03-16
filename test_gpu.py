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
from tensorflow.keras.optimizers import Adam

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
        
        # Configure memory growth to avo
