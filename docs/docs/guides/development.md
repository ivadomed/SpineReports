---
layout: default
title: Development Setup
parent: Documentation
grand_parent: Documentation
nav_order: 6
---

# Development Setup

This guide explains how to set up a development environment for SpineReport.

## Prerequisites

- Python 3.10 or higher
- Git
- pip or conda

## Clone the Repository

```bash
git clone https://github.com/ivadomed/SpineReport.git
cd SpineReport
```

## Create a Virtual Environment

### Using venv

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Using conda

```bash
conda create -n spinereport-dev python=3.10
conda activate spinereport-dev
```

## Install Dependencies

### Development Installation

Install SpineReport in editable mode with development dependencies:

```bash
pip install -e ".[dev]"
```

### Install Optional Dependencies

For full development setup:

```bash
# GPU support (optional)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Documentation
pip install sphinx sphinx-rtd-theme

# Testing and linting
pip install pytest pytest-cov flake8 black isort
```

## Project Structure

```
SpineReport/
├── spinereport/           # Main package
│   ├── __init__.py
│   ├── spinereport.py     # Main module
│   ├── plot_by_group.py   # Plotting module
│   ├── resources/         # Resources and images
│   └── utils/             # Utility modules
│       ├── image.py
│       ├── measure_seg.py
│       ├── utils.py
│       └── generate_reports.py
├── docs/                  # Documentation (Jekyll)
├── tests/                 # Test files
├── pyproject.toml         # Project configuration
└── README.md
```

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=spinereport

# Run specific test file
pytest tests/test_utils.py

# Run with verbose output
pytest -v
```

## Code Quality

### Formatting

Format code with Black:

```bash
black spinereport/
```

### Linting

Check code style with flake8:

```bash
flake8 spinereport/
```

### Import Sorting

Sort imports with isort:

```bash
isort spinereport/
```

### Run All Quality Checks

```bash
# Format
black spinereport/ tests/

# Lint
flake8 spinereport/ tests/

# Sort imports
isort spinereport/ tests/

# Run tests
pytest
```

## Building Documentation

Build the documentation locally:

```bash
cd docs
bundle install
bundle exec jekyll serve
```

The documentation will be available at `http://localhost:4000`.

## Making Changes

1. Create a feature branch:
   ```bash
   git checkout -b feature/my-feature
   ```

2. Make your changes
3. Run tests and quality checks
4. Commit your changes
5. Push and open a pull request

## Debugging

### Using print statements

```python
def my_function(x):
    print(f"Debug: x = {x}")
    return x * 2
```

### Using Python debugger

```python
import pdb

def my_function(x):
    pdb.set_trace()  # Execution will pause here
    return x * 2
```

### Using pytest with debugging

```bash
# Drop into debugger on failure
pytest --pdb

# Drop into debugger at start
pytest --trace
```

## Common Tasks

### Add a New Module

1. Create file in `spinereport/`
2. Import in `spinereport/__init__.py`
3. Add tests in `tests/`
4. Add documentation

### Update Dependencies

Edit `pyproject.toml`:

```toml
[project]
dependencies = [
    "new-package>=1.0",
    "existing-package>=2.0",
]
```

Then reinstall:

```bash
pip install -e ".[dev]"
```

## Getting Help

- Check [Contributing Guide](contributing.md)
- Open an issue on GitHub
- Ask in a discussion

Happy developing! 🚀
