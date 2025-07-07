# Llama3-Mini

A minimal implementation of Llama-3 with MoE (Mixture of Experts) support.

## Project Structure

```
src/llama3_mini/
├── models/          # Model implementations
├── data/           # Data processing
├── training/       # Training utilities
├── evaluation/     # Evaluation scripts
└── utils/          # Utility functions
```

## Setup

1. Activate environment: `source activate.sh`
2. Install dependencies: `uv sync --dev`
3. Prepare data: `python scripts/prepare_data.py`

## Models

- **Dense**: ~160M parameters
- **MoE**: ~160M activated, ~320M total (4 experts)
