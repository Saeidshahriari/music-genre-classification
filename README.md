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
- Elie
- Mykhailo

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

## Results

We compared three approaches on the GTZAN Genre Collection (999 tracks usable, 10 genres). Stratified split: 559 train / 140 validation / 300 test.

### Final Test Accuracy

| Approach | Features | Classifier | Test Accuracy |
|---|---|---|---|
| 1a | MFCC (20 coefficients, mean-pooled) | SVM (RBF kernel) | 68.67% |
| 1b | MFCC (20 coefficients, mean-pooled) | MLP (2 hidden layers) | 66.00% |
| **2** | **Mel-spectrogram (128 mel bands)** | **CNN (3 conv blocks)** | **82.33%** |

The CNN on mel-spectrograms outperformed handcrafted-feature baselines by **+13.66 percentage points**, consistent with findings in [van den Oord et al. (2013)](https://papers.nips.cc/paper/2013/hash/b3ba8f1bee1238a2f37603d90b58898d-Abstract.html) and [Choi et al. (2017)](https://arxiv.org/abs/1609.04243), which showed deep representations outperform handcrafted audio features. Our 82.33% accuracy falls within the 80-85% range cited in the VUB project guide for CNN/CRNN on GTZAN.

### Per-Genre Performance (CNN, Approach 2)

| Genre | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|
| metal | 0.91 | 0.97 | 0.94 | 96.7% |
| classical | 0.93 | 0.93 | 0.93 | 93.3% |
| hiphop | 0.80 | 0.93 | 0.86 | 93.3% |
| blues | 0.82 | 0.90 | 0.86 | 90.0% |
| jazz | 0.87 | 0.87 | 0.87 | 86.7% |
| pop | 0.81 | 0.83 | 0.82 | 83.3% |
| country | 0.71 | 0.73 | 0.72 | 73.3% |
| reggae | 0.88 | 0.70 | 0.78 | 70.0% |
| rock | 0.68 | 0.70 | 0.69 | 70.0% |
| disco | 0.87 | 0.67 | 0.75 | 66.7% |
| **Macro avg** | **0.83** | **0.82** | **0.82** | **82.3%** |

### Key Findings

**Genres with sharp acoustic identity are easier to classify.** Metal (96.7%) and classical (93.3%) have distinctive timbral signatures (heavy distortion vs. orchestral instrumentation) that the CNN learned reliably.

**Genres with overlapping characteristics confuse the model.** The most frequent confusions on the test set:
- disco → pop (4 tracks)
- country → blues (4 tracks)
- reggae → hiphop (3 tracks)
- jazz → country (3 tracks)

These are musically reasonable confusions. Disco and pop share rhythmic patterns and production style; country and blues share instrumentation and chord progressions. This matches observations in [Sturm (2013)](https://arxiv.org/abs/1306.1461), who documented that GTZAN genre boundaries are inherently fuzzy.

**Rock and disco were the weakest genres** (70.0% and 66.7%). Rock acts as a "catch-all" genre that overlaps with metal, blues, and pop. Disco's characteristic 4-on-the-floor pattern and instrumentation overlap heavily with pop production from similar eras.

### Reproducibility Notes

- **Data:** 1 corrupted file in original GTZAN distribution (`jazz.00054.wav`, "Format not recognised") was skipped, leaving 999 tracks. This is a [documented issue](https://arxiv.org/abs/1306.1461) with the dataset.
- **Hardware:** CPU only (no GPU). Total training time ≈ X minutes.
- **Random seeds:** fixed across all approaches for reproducibility.
- **Best CNN epoch:** 47 (validation accuracy 85.7%), used for test set evaluation.

Full per-experiment details and confusion matrices are saved to `approach1_results.json` (when re-running locally).

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for branch naming, commit conventions, and PR rules.

## License

MIT, see [`LICENSE`](LICENSE).
