#!/bin/bash
# TensorFlow GPU Benchmark with Direct Display
# This script runs the benchmark tools and displays results directly in the console

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Display header
echo -e "${GREEN}=====================================================${NC}"
echo -e "${GREEN}            TensorFlow GPU Benchmark Suite           ${NC}"
echo -e "${GREEN}=====================================================${NC}"

# Check if script exists
if [ ! -f "/app/tensorflow_benchmark.py" ]; then
    echo -e "${RED}Error: Benchmark script not found${NC}"
    echo "Copying script to the right location..."
    # You can copy the script here if needed
    exit 1
fi

# Make script executable if needed
chmod +x /app/tensorflow_benchmark.py

# Run the benchmark
echo -e "\n${YELLOW}Running TensorFlow GPU benchmark...${NC}"
python3 /app/tensorflow_benchmark.py

echo -e "\n${GREEN}=====================================================${NC}"
echo -e "Benchmark complete!"
echo -e "${GREEN}=====================================================${NC}"