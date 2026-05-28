---
layout: default
title: Contributing
parent: Documentation
grand_parent: Documentation
nav_order: 5
---

# Contributing to SpineReport

We welcome contributions! This guide explains how to contribute to the SpineReport project.

## Getting Started

1. **Fork** the repository on GitHub
2. **Clone** your fork locally
3. **Create** a new branch for your feature or fix

```bash
git clone https://github.com/YOUR-USERNAME/SpineReport.git
cd SpineReport
git checkout -b feature/your-feature-name
```

## Development Setup

See [Development Setup](development.md) for instructions on setting up your development environment.

## Making Changes

1. Make your changes to the code
2. Add or update tests as needed
3. Run the test suite to ensure everything passes
4. Update documentation if necessary

## Code Style

SpineReport follows PEP 8 conventions. Please:

- Use clear, descriptive variable names
- Add docstrings to functions and classes
- Keep functions small and focused
- Add type hints where possible

Example:
```python
def compute_measurement(image: np.ndarray, mask: np.ndarray) -> float:
    """
    Compute a measurement from image and mask.
    
    Parameters
    ----------
    image : np.ndarray
        Input image array
    mask : np.ndarray
        Segmentation mask
        
    Returns
    -------
    float
        Computed measurement value
    """
    pass
```

## Testing

Before submitting a pull request, ensure all tests pass:

```bash
pytest tests/
```

Add tests for new features:

```bash
# Create a test file
touch tests/test_new_feature.py

# Run just your new tests
pytest tests/test_new_feature.py -v
```

## Submitting Changes

1. **Commit** your changes with clear, descriptive messages:
   ```bash
   git commit -m "Add feature: description of what was added"
   ```

2. **Push** to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

3. **Open a Pull Request** on GitHub with:
   - A clear title
   - Description of changes
   - Reference to any related issues
   - Screenshots for UI changes

## Pull Request Guidelines

- Keep PRs focused on a single feature or fix
- Include tests for new functionality
- Update documentation
- Ensure CI checks pass
- Be respectful and constructive in discussions

## Reporting Issues

Found a bug? Please [open an issue](https://github.com/ivadomed/SpineReport/issues) with:

- Clear title and description
- Steps to reproduce
- Expected vs actual behavior
- Python and SpineReport versions
- Environment details

## Documentation

Help improve the docs! You can:

- Fix typos or unclear explanations
- Add examples or tutorials
- Translate documentation
- Improve API documentation

Documentation files are in the `docs/` directory as Markdown files.

## Code Review

Your PR will be reviewed by maintainers. Please:

- Respond to feedback promptly
- Request re-review after making changes
- Discuss any concerns respectfully

## Questions?

- Check existing [issues](https://github.com/ivadomed/SpineReport/issues)
- Ask in a new issue with the `question` label
- Email the maintainers

Thank you for contributing to SpineReport! 🎉
