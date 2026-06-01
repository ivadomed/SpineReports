# Documentation Setup for SpineReport

This directory contains the source files for SpineReport documentation built with [Just the Docs](https://just-the-docs.com) and Jekyll.

## Building Documentation Locally

### Prerequisites

- Ruby >= 2.7
- Bundler

### Installation

```bash
cd docs
bundle install
```

### Building

```bash
# Build the site
bundle exec jekyll build

# Serve locally (at http://localhost:4000)
bundle exec jekyll serve
```

The built site will be in `_site/` directory.

## Publishing to GitHub Pages

The documentation is automatically published to GitHub Pages via GitHub Actions when changes are pushed to the `main` branch.

### Manual Configuration (if needed)

1. Go to your repository Settings → Pages
2. Set Build and deployment → Source to "GitHub Actions"
3. The `pages.yml` workflow will automatically build and deploy the site

### Site URL

The documentation will be published at:
```
https://ivadomed.github.io/SpineReport
```

## Project Structure

```
docs/
├── _config.yml          # Jekyll configuration
├── Gemfile              # Ruby dependencies
├── index.md             # Home page
├── docs/                # Documentation pages
│   ├── installation.md
│   └── getting-started.md
└── _site/               # Built output (auto-generated)
```

## Editing Documentation

All documentation is written in Markdown with YAML frontmatter. Each file should start with:

```yaml
---
layout: default
title: Page Title
parent: Parent Page (optional)
nav_order: 1
---
```

### Creating New Pages

1. Create a new `.md` file
2. Add YAML frontmatter with title and navigation info
3. Write content in Markdown
4. Commit and push - it will auto-publish

### Navigation Structure

Use `parent` and `nav_order` to structure navigation:

```yaml
---
layout: default
title: My Guide
parent: Documentation
nav_order: 3
---
```

## Customization

To customize the theme, edit `_config.yml`:

- `title`: Site title
- `description`: Site description
- `url`: Site URL (important for GitHub Pages)
- `color_scheme`: Theme color (nil, light, dark)
- `search_enabled`: Enable/disable search

See [Just the Docs Configuration](https://just-the-docs.com/docs/configuration/) for all options.

## Resources

- [Just the Docs Documentation](https://just-the-docs.com)
- [Jekyll Documentation](https://jekyllrb.com/docs/)
- [Markdown Guide](https://www.markdownguide.org/)

## Troubleshooting

### Port already in use

```bash
bundle exec jekyll serve --port 4001
```

### Changes not showing up

```bash
# Clear and rebuild
bundle exec jekyll clean
bundle exec jekyll serve
```

### Ruby dependency issues

```bash
# Update dependencies
bundle update
bundle exec jekyll serve
```

## More Information

- See main [README.md](../README.md)
- GitHub Pages docs: https://docs.github.com/en/pages
- Just the Docs: https://just-the-docs.com
