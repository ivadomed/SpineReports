---
layout: default
title: API Reference
parent: Documentation
nav_order: 3
---

# API Reference

## Main Modules

### spinereport.spinereport

Main module for report generation.

#### `generate_report()`

```python
generate_report(
    input_image: str,
    output_dir: str,
    control_group_data: Optional[str] = None,
    patient_info: Optional[dict] = None
) -> str
```

Generate a radiological report for a spinal MRI image.

**Parameters:**
- `input_image` (str): Path to the MRI image file (.nii or .nii.gz)
- `output_dir` (str): Directory where the report will be saved
- `control_group_data` (str, optional): Path to control group statistics
- `patient_info` (dict, optional): Patient metadata dictionary

**Returns:**
- str: Path to the generated PDF report

**Example:**
```python
from spinereport.spinereport import generate_report

report = generate_report(
    input_image="patient_mri.nii.gz",
    output_dir="./reports",
    patient_info={"name": "John Doe", "age": 45}
)
```

### spinereport.utils

Utility functions for image processing and analysis.

#### Image Processing

```python
from spinereport.utils.image import load_nifti, normalize_image

# Load NIFTI image
image_data = load_nifti("image.nii.gz")

# Normalize image intensities
normalized = normalize_image(image_data)
```

#### Measurement Functions

```python
from spinereport.utils.measure_seg import (
    compute_canal_diameter,
    compute_vertebrae_height,
    compute_disc_height
)

# Compute spinal measurements
canal_diameter = compute_canal_diameter(segmentation_data)
vertebrae_height = compute_vertebrae_height(segmentation_data)
disc_height = compute_disc_height(segmentation_data)
```

#### Report Generation

```python
from spinereport.utils.generate_reports import create_pdf_report

# Create a PDF report
create_pdf_report(
    measurements=measurements_dict,
    output_path="report.pdf",
    control_group_stats=control_stats
)
```

## Data Structures

### Measurement Dictionary

```python
measurements = {
    'canal_diameter': [10.2, 10.5, 10.1, 9.8],  # mm
    'vertebrae_height': [25.3, 25.1, 24.9],  # mm
    'disc_height': [8.5, 8.3, 8.2],  # mm
    'foraminal_width': [12.5, 12.3, 12.1]  # mm
}
```

### Patient Info Dictionary

```python
patient_info = {
    'name': 'John Doe',
    'age': 45,
    'sex': 'M',
    'mri_date': '2024-01-15',
    'indication': 'Back pain'
}
```

## Configuration

SpineReport can be configured via environment variables or configuration files.

### Environment Variables

```bash
# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Set number of threads
export OMP_NUM_THREADS=4
```

### Configuration File

Create a `spinereport_config.yaml`:

```yaml
segmentation:
  model: "nnunet"
  cuda: true

output:
  format: "pdf"
  include_statistics: true
```

## Exception Handling

```python
from spinereport.spinereport import SpineReportError

try:
    report = generate_report(...)
except SpineReportError as e:
    print(f"Error generating report: {e}")
except FileNotFoundError:
    print("Input file not found")
```

## See Also

- [Quick Start Guide](getting-started.md)
- [Installation Instructions](installation.md)
- [GitHub Repository](https://github.com/ivadomed/SpineReport)
