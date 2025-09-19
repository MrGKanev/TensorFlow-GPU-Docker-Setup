#!/bin/bash
# entrypoint.sh - Simple, robust container entrypoint

set -euo pipefail

# Colors
GREEN="\033[0;32m"
YELLOW="\033[1;33m"
RED="\033[0;31m"
BLUE="\033[0;34m"
NC="\033[0m"

# Header
echo -e "${GREEN}TensorFlow GPU Container${NC}"
echo "=================================================="

# Quick system info
echo -e "${BLUE}ℹ Container started at: $(date)${NC}"
echo -e "${BLUE}ℹ Python: $(python --version 2>&1)${NC}"

# GPU check
echo -e "${YELLOW} Checking GPU availability...${NC}"

if command -v nvidia-smi >/dev/null 2>&1; then
    if nvidia-smi >/dev/null 2>&1; then
        echo -e "${GREEN} NVIDIA SMI available${NC}"
        GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l || echo 0)
        echo -e "${BLUE} Hardware GPUs detected: $GPU_COUNT${NC}"
    else
        echo -e "${YELLOW}⚠️  nvidia-smi failed to run${NC}"
    fi
else
    echo -e "${YELLOW} nvidia-smi not available${NC}"
fi

# TensorFlow GPU check
echo -e "${YELLOW} Checking TensorFlow GPU support...${NC}"

TF_GPU_CHECK=$(python -c "
import tensorflow as tf
try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(f'TF_GPUS:{len(gpus)}')
    if gpus:
        print('TF_GPU_SUPPORT:True')
    else:
        print('TF_GPU_SUPPORT:False')
except Exception as e:
    print(f'TF_ERROR:{e}')
" 2>/dev/null || echo "TF_ERROR:Failed to import TensorFlow")

if echo "$TF_GPU_CHECK" | grep -q "TF_ERROR:"; then
    echo -e "${RED} TensorFlow GPU check failed${NC}"
else
    TF_GPUS=$(echo "$TF_GPU_CHECK" | grep "TF_GPUS:" | cut -d: -f2)
    if [[ "$TF_GPUS" -gt 0 ]]; then
        echo -e "${GREEN} TensorFlow GPU support verified${NC}"
        echo -e "${BLUE}ℹ  TensorFlow detected $TF_GPUS GPU(s)${NC}"
    else
        echo -e "${YELLOW}  TensorFlow running in CPU-only mode${NC}"
    fi
fi

# Handle commands
case "${1:-}" in
    "--benchmark"|"benchmark")
        echo -e "${BLUE} Running benchmark...${NC}"
        if [[ -f "/app/tf_benchmark.py" ]]; then
            exec python /app/tf_benchmark.py --small
        else
            echo -e "${RED} Benchmark script not found${NC}"
            exit 1
        fi
        ;;

    "--check-gpu"|"check-gpu")
        echo -e "${BLUE}🔍 Running detailed GPU check...${NC}"
        if [[ -f "/app/check_gpu.py" ]]; then
            exec python /app/check_gpu.py
        else
            echo -e "${RED} GPU check script not found${NC}"
            exit 1
        fi
        ;;

    "--jupyter"|"jupyter")
        echo -e "${BLUE}Starting Jupyter Lab...${NC}"
        exec jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --LabApp.token=development
        ;;

    "--help"|"help")
        echo "=================================================="
        echo "Available commands:"
        echo "  --benchmark       : Run GPU benchmark"
        echo "  --check-gpu       : Detailed GPU diagnostics"
        echo "  --jupyter         : Start Jupyter Lab"
        echo "  --help            : Show this help"
        echo "=================================================="
        exit 0
        ;;

    "")
        echo "=================================================="
        echo -e "${GREEN}Container ready!${NC}"
        echo "Available commands: --benchmark, --check-gpu, --jupyter, --help"
        echo "=================================================="
        exec bash
        ;;

    *)
        exec "$@"
        ;;
esac