---
layout: default
title: Quick Start
parent: Documentation
nav_order: 3
---

# Quick Start Guide

## Input Format

SpineReport expects:
- **Data**: NIFTI format MRI images (`.nii` or `.nii.gz`)
- **File structure**: Each subject's data must be in a flatted directory structure (i.e. all images in the same directory).

## Segmenting the spinal anatomy

Before generating a report, you need to segment the spinal anatomy from your MRI data. For now we recommand using [TotalSpineSeg](https://github.com/neuropoly/totalspineseg), which is a robust state-of-the-art deep learning model. Once installed, you can run the segmentation with the following command:

```bash
totalspineseg <input_directory> <output_directory> --iso
```

**Example:**
```bash
totalspineseg .data/raw .data/out --iso
```

> If you have two groups (e.g. test and control), you need to run the segmentation separately for each group and save the outputs in separate directories.

## Command Line Interface

SpineReport provides two main command-line tools:

### 1. Generate Reports

This command generates structured radiological reports from segmented spinal anatomy:

```bash
spinereport -t <test_segmentation_directory> -c <control_segmentation_directory> -o <report_output_directory>
```

**Parameters:**
- `-t` (or `--test-dir`): Path to directory containing the test group segmentations
- `-c` (or `--control-dir`): Path to directory containing the control group segmentations (you can also specify the path to the test group if you don't have a separate control group)
- `-o` (or `--ofolder`): Path where reports will be saved

**Example:**
```bash
spinereport -t ./data/out_test -c ./data/out_control -o ./reports
```

### 2. Plot By Group

This command generates comparison plots by group (age and sex) from the segmentation and demographics data:

```bash
spinereport_plot_by_group -i <segmentation_directory> -d <demographics_file> -o <output_directory>
```

**Parameters:**
- `-i` (or `--input-dir`): Path to directory containing the segmentation data
- `-d` (or `--demographics-file`): Path to CSV file containing demographic information with columns corresponding to the "participant_id", the "sex" and "age" columns.
- `-o` (or `--ofolder`): Path where plots will be saved

**Example:**
```bash
spinereport_plot_by_group -i ./data/out_test -d ./data/demographics.csv -o ./plots
```

## Output

SpineReport generates:
- **PDF Reports**: Radiological subjects' specific reports with comparisons against control group
- **Data CSV**: Raw measurement data in CSV format

## Next Steps

- [Review detailed documentation](../docs/){: .btn }
- [Visit GitHub repository](https://github.com/ivadomed/SpineReport){: .btn }

**Need help?** Check the [documentation](docs/index/) or open an [issue](https://github.com/ivadomed/SpineReport/issues).
