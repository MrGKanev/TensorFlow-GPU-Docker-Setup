#!/bin/bash
# TensorFlow GPU Benchmark Runner
# This script runs the benchmark tools and reports results

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Create results directory
mkdir -p /app/benchmark_results

# Display header
echo -e "${GREEN}=====================================================${NC}"
echo -e "${GREEN}          TensorFlow GPU Benchmark Suite             ${NC}"
echo -e "${GREEN}=====================================================${NC}"

# Check for GPU availability
echo -e "\nChecking for GPU availability..."
GPU_COUNT=$(python3 -c "import tensorflow as tf; print(len(tf.config.list_physical_devices('GPU')))")

if [ "$GPU_COUNT" -eq "0" ]; then
    echo -e "${RED}❌ No GPU detected. Tests will run on CPU only.${NC}"
    echo -e "${YELLOW}Possible causes:${NC}"
    echo "  - NVIDIA drivers not installed on host"
    echo "  - Container not started with --gpus all flag"
    echo "  - NVIDIA Container Toolkit not installed"
    echo -e "  - Incompatible GPU\n"
    echo "Run:"
    echo "  nvidia-smi        (on host to check drivers)"
    echo "  docker run --gpus all --rm nvidia/cuda:11.6.2-base-ubuntu20.04 nvidia-smi"
    echo "                    (to check NVIDIA Container Toolkit)"
else
    echo -e "${GREEN}✅ Found $GPU_COUNT GPU(s).${NC}"
fi

# Ask which benchmark to run
echo -e "\nSelect benchmark to run:"
echo "  1) Quick benchmark (1-2 minutes)"
echo "  2) Full benchmark suite (10-15 minutes)"
echo "  3) Both benchmarks"
echo "  4) Exit"
read -p "Enter choice [1-4]: " choice

case $choice in
    1)
        echo -e "\n${YELLOW}Running quick benchmark...${NC}"
        python3 /app/tensorflow_benchmark_quick.py
        ;;
    2)
        echo -e "\n${YELLOW}Running full benchmark suite...${NC}"
        python3 /app/tensorflow_benchmark.py
        ;;
    3)
        echo -e "\n${YELLOW}Running quick benchmark...${NC}"
        python3 /app/tensorflow_benchmark_quick.py
        
        echo -e "\n${YELLOW}Running full benchmark suite...${NC}"
        python3 /app/tensorflow_benchmark.py
        ;;
    4)
        echo "Exiting."
        exit 0
        ;;
    *)
        echo -e "${RED}Invalid choice. Exiting.${NC}"
        exit 1
        ;;
esac

# Display results location
echo -e "\n${GREEN}=====================================================${NC}"
echo -e "Benchmark results saved to: /app/benchmark_results/"
echo -e "${GREEN}=====================================================${NC}"