#!/bin/bash
# Activate llama3-mini environment

# Source environment variables
source "$(dirname "$0")/.env"

# Activate virtual environment
source "$LLAMA3_STORE/venv/bin/activate"

echo "Llama3-Mini environment activated!"
echo "Project: $LLAMA3_PROJECT"
echo "Storage: $LLAMA3_STORE"
echo "Python: $(python --version)"
echo "UV Cache: $UV_CACHE_DIR"

# Test installations
python -c "
import sys
print(f'Python executable: {sys.executable}')

try:
    import torch
    print(f'PyTorch: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'CUDA version: {torch.version.cuda}')
        print(f'GPU: {torch.cuda.get_device_name(0)}')
        print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
except ImportError:
    print('PyTorch: Not installed yet')

try:
    import flash_attn
    print('Flash Attention: Available')
except ImportError:
    print('Flash Attention: Not available')
"
