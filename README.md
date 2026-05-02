# Music Genre Classification

Deep learning project for the *Machine Learning and Big Data Processing* course at VUB, 2025-2026.

[![CI](https://github.com/Saeidshahriari/music-genre-classification/actions/workflows/ci.yml/badge.svg)](https://github.com/Saeidshahriari/music-genre-classification/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

## Overview

Classify audio tracks into musical genres using deep learning on log-mel spectrograms. Project goals:
1. Build a reproducible training pipeline
2. Compare a CNN baseline against a transformer model
3. Track experiments and serve the best model

## Team
- Saeid Shahriari ([@Saeidshahriari](https://github.com/Saeidshahriari))
- Elie [last name] ([@handle])
- [Third member] ([@handle])

## Quickstart

### Requirements
- Python 3.11+
- [uv](https://docs.astral.sh/uv/) for dependency management
- (Optional) Docker

### Setup
```bash
git clone git@github.com:Saeidshahriari/music-genre-classification.git
cd music-genre-classification
uv sync
uv run pre-commit install
```

### Run tests
```bash
make test
```

### Train
```bash
uv run python scripts/train.py --config configs/default.yaml
```

## Project structure

```
src/music_genre/   library code (importable)
tests/             unit and integration tests
notebooks/         exploratory work, not for production
configs/           experiment configuration (YAML)
scripts/           entry points (training, evaluation)
docs/              architecture, design decisions
```

## Data

The dataset is not stored in this repository (see `.gitignore`). Download instructions are in [`docs/data.md`](docs/data.md).

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for branch naming, commit conventions, and PR rules.

## License

MIT, see [`LICENSE`](LICENSE).