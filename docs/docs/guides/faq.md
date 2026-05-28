---
layout: default
title: FAQ
parent: Documentation
grand_parent: Documentation
nav_order: 4
---

# Frequently Asked Questions

## Installation & Setup

### Q: What Python versions does SpineReport support?

A: SpineReport requires **Python 3.10 or higher**. We recommend using Python 3.10 or 3.11 for best compatibility.

### Q: Can I use SpineReport on Windows?

A: Yes, SpineReport works on Windows, macOS, and Linux. Install using pip as described in the [Installation Guide](../installation.md).

### Q: Do I need GPU to use SpineReport?

A: GPU is optional but recommended for faster processing. SpineReport can run on CPU, but processing will be slower. Install CUDA-enabled PyTorch for GPU support.

### Q: How much disk space do I need?

A: SpineReport requires approximately 2-3 GB for the model weights and dependencies. Additional space needed depends on your image size and number of analyses.

## Usage

### Q: What image formats does SpineReport accept?

A: SpineReport accepts **NIFTI format** images (`.nii` or `.nii.gz`). If you have DICOM images, convert them to NIFTI first using tools like dcm2niix.

### Q: How long does it take to generate a report?

A: Report generation typically takes 5-15 minutes depending on:
- Image resolution and size
- Available computational resources
- Number of control group comparisons

### Q: Can I process multiple patients at once?

A: Yes, use batch processing:

```bash
spinereport -i ./input_directory -o ./output_directory
```

SpineReport will process all images in the input directory.

### Q: How do I use my own control group data?

A: Provide control group statistics in CSV format:

```bash
spinereport -i ./test_cases -o ./reports -control-group ./control_stats.csv
```

The CSV should contain measurement columns (canal_diameter, vertebrae_height, etc.).

## Reports & Output

### Q: What information is included in the generated reports?

A: Reports include:
- Patient demographics and metadata
- Morphometric measurements (canal diameter, vertebrae height, etc.)
- Comparative graphs with control group statistics
- Statistical summaries and percentiles
- Clinical interpretation notes

### Q: Can I customize the report template?

A: Basic customization is available through the Python API. For advanced customization, modify the reporting templates in the source code.

### Q: What format are the output reports?

A: Reports are generated as PDF files, optimized for printing and sharing. Raw measurement data is also exported as CSV.

### Q: Can I batch process and get CSV output only?

A: Yes, use the Python API:

```python
from spinereport.utils.measure_seg import compute_measurements

measurements = compute_measurements(image_path)
# measurements is a dictionary ready to save as CSV
```

## Troubleshooting

### Q: I get a "CUDA out of memory" error

A: Either:
1. Process smaller images or image patches
2. Reduce batch size in configuration
3. Use CPU instead: `export CUDA_VISIBLE_DEVICES=""`
4. Use a GPU with more VRAM

### Q: The segmentation looks incorrect

A: Try:
1. Check image quality and orientation
2. Ensure proper NIFTI header information
3. Verify image is in the correct format
4. Try preprocessing the image (normalization, cropping)

### Q: I'm getting "File not found" errors

A: Ensure:
1. Input file paths are correct
2. Use absolute paths if possible
3. Check file permissions
4. Verify the file extension is `.nii` or `.nii.gz`

### Q: Installation fails with dependency conflicts

A: Try:
```bash
pip install --upgrade pip setuptools
pip install --force-reinstall spinereport
```

Or use conda for better dependency resolution:
```bash
conda install -c conda-forge spinereport
```

## Data & Privacy

### Q: Is patient data stored anywhere?

A: No, SpineReport only processes files locally on your machine. No data is uploaded or stored externally.

### Q: Can I use SpineReport in a clinical setting?

A: SpineReport is a research tool. For clinical use, consult your institution's guidelines and ensure appropriate validation and regulatory compliance.

### Q: How do I securely process sensitive patient data?

A: Recommendations:
1. Use isolated secure environment
2. Process locally without network access
3. Implement proper access controls
4. Follow HIPAA/GDPR guidelines as applicable
5. Use de-identified data for training/testing

## Getting Help

### Q: Where can I report bugs?

A: Report issues on [GitHub Issues](https://github.com/ivadomed/SpineReport/issues).

### Q: How do I request a new feature?

A: Open a [GitHub Issue](https://github.com/ivadomed/SpineReport/issues) with the "enhancement" label.

### Q: Where can I find more examples?

A: Check the [Quick Start Guide](../getting-started.md) and [API Reference](../api.md).

### Q: Can I contribute to SpineReport?

A: Yes! See the [Contributing Guide](contributing.md).

---

**Can't find what you're looking for?** Open an [issue](https://github.com/ivadomed/SpineReport/issues) or email the maintainers.
