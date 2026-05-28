---
layout: default
title: Quick Start
parent: Documentation
nav_order: 2
---

# Quick Start Guide

## Basic Usage

### Command Line Interface

SpineReport provides two main command-line tools:

#### 1. Generate Reports

```bash
spinereport --input <input_directory> --output <output_directory>
```

**Parameters:**
- `--input` (or `-i`): Path to directory containing input MRI images
- `--output` (or `-o`): Path where reports will be saved
- `--control-group` (optional): Path to control group data for comparison

**Example:**
```bash
spinereport -i ./data/test_cases -o ./reports -control-group ./data/control
```

#### 2. Plot By Group

```bash
spinereport_plot_by_group --input <input_file> --output <output_directory>
```

This tool creates comparison plots of measurements across different groups.

## Input Format

SpineReport expects:
- **Input**: NIFTI format MRI images (`.nii` or `.nii.gz`)
- **File structure**: Each subject's data can be in a separate directory or combined

## Output

SpineReport generates:
- **PDF Reports**: Professional radiological reports with measurements and comparisons
- **Data CSV**: Raw measurement data in CSV format
- **Plots**: Visualization graphs comparing test and control groups

## Python API

You can also use SpineReport programmatically:

```python
from spinereport import generate_report

# Generate a single report
report_path = generate_report(
    input_image="path/to/image.nii.gz",
    output_dir="path/to/output",
    control_group_data="path/to/control_group_stats.csv"
)

print(f"Report saved to: {report_path}")
```

## Example Workflow

```bash
# 1. Create directories
mkdir -p data/test_cases data/control output/reports

# 2. Place your MRI images
# Copy test case images to data/test_cases/
# Copy control group images to data/control/

# 3. Generate reports
spinereport -i data/test_cases -o output/reports -control-group data/control

# 4. View results
# Open the generated PDF files in output/reports/
```

## Report Contents

Each generated report includes:

- **Patient Information**: Demographics and metadata
- **Morphometric Measurements**: Spinal canal, vertebrae, discs measurements
- **Comparative Analysis**: Graphs comparing to control group
- **Clinical Interpretation**: Statistical summaries

## Next Steps

- [Review detailed documentation](../docs/){: .btn }
- [Check API reference](../api/){: .btn }
- [Visit GitHub repository](https://github.com/ivadomed/SpineReport){: .btn }
