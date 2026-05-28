---
layout: default
title: Installation
parent: Documentation
nav_order: 1
---

# Installation

## Requirements

- Python >= 3.10
- pip >= 23
- setuptools >= 67

## Setup Instructions

### 1. Create a Virtual Environment

We recommend using a virtual environment to isolate dependencies.

**Using venv:**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**Using conda:**
```bash
conda create -n spinereport python=3.10
conda activate spinereport
```

### 2. Install SpineReport

**From PyPI (recommended):**
```bash
pip install spinereport
```

**From source:**
```bash
git clone https://github.com/ivadomed/SpineReport.git
cd SpineReport
pip install -e .
```

### 3. Verify Installation

Test the installation by checking the help:
```bash
spinereport --help
```

## Dependencies

SpineReport requires the following Python packages:

- **opencv-python** - Image processing

And will automatically install all required dependencies including:
- nnU-Net for segmentation
- NumPy for numerical operations
- scikit-image for image analysis
- matplotlib for visualization
- pandas for data handling

## Troubleshooting

### CUDA/GPU Support

If you want to use GPU acceleration with CUDA, install the GPU version:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Installation Issues

If you encounter issues during installation, try:

1. Update pip and setuptools:
   ```bash
   pip install --upgrade pip setuptools
   ```

2. Install in development mode:
   ```bash
   pip install -e ".[dev]"
   ```

3. Check your Python version:
   ```bash
   python --version  # Should be >= 3.10
   ```

For additional support, please [open an issue](https://github.com/ivadomed/SpineReport/issues).
