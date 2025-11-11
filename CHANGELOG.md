# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2025-11-10

### Added

#### Data Loading
- **DETRIS Zenodo integration**: Download real frequency-domain THz spectra from Zenodo (DOI: 10.5281/zenodo.5079558)
- **Hermitian IFFT implementation**: Convert frequency-domain spectra to real-valued time-domain waveforms using conjugate symmetry
- **Configurable data source**: `USE_DETRIS` flag in notebook to switch between real and synthetic data
- **Deterministic seeding**: `numpy.random.default_rng(42)` and `PYTHONHASHSEED=0` for reproducibility
- **Graceful fallback**: Automatic switch to synthetic data if DETRIS download fails

#### Code Organization
- **Modular architecture**: Split monolithic notebook into reusable modules:
  - `src/data.py`: Data loading (DETRIS + synthetic)
  - `src/preprocess.py`: Baseline correction, SavGol smoothing, FFT
  - `src/features.py`: Peak detection, spectral moments
  - `src/model.py`: Random Forest training, artifact management
  - `src/visualise.py`: Plotly figure builders with PNG export
  - `src/predict.py`: CLI for inference
- **Type hints**: Added type annotations for better IDE support

#### Testing
- **Comprehensive test suite**:
  - `tests/test_data_loader.py`: Data loading, shapes, NaN/Inf checks, online/offline modes
  - `tests/test_features.py`: Feature extraction, determinism, fixed-length vectors
  - `tests/test_model.py`: Training, accuracy benchmarks, reproducibility
- **Offline compatibility**: Tests skip DETRIS download if offline (pytest markers)

#### CI/CD
- **GitHub Actions workflow** (`.github/workflows/ci.yaml`):
  - Pip package caching (keyed on `requirements.txt`)
  - Deterministic execution (`PYTHONHASHSEED=0`)
  - Linting (black, isort, ruff)
  - Test execution (pytest)
  - Headless notebook execution (papermill)
  - Artifact upload (HTML, PNG, models)

#### Code Quality
- **Pre-commit hooks** (`.pre-commit-config.yaml`):
  - Trailing whitespace removal
  - End-of-file fixer
  - YAML/JSON validation
  - Black formatting
  - isort import sorting
  - Ruff linting
- **Configuration files**:
  - `pyproject.toml`: Black, isort, ruff, pytest settings
  - `.gitignore`: Comprehensive ignore rules

#### Visualisation
- **PNG export**: All Plotly figures now export to static PNG via kaleido
- **Artifact organisation**: All outputs saved to `artifacts/` directory:
  - `time_domain.html` / `.png`
  - `spectra.html` / `.png`
  - `pca_scatter.html` / `.png`
  - `feature_importances.html` / `.png`
  - `model.joblib`, `scaler.joblib`, `pca.joblib`
  - `prediction.json`

#### Documentation
- **Expanded README**:
  - Release notes section
  - Mermaid pipeline diagram
  - Data provenance (Zenodo DOI, CC-BY 4.0 licence)
  - Hermitian IFFT explanation
  - Reproducibility notes
  - CI badges
  - Swap data sources guide
  - Embedded PNG screenshots
- **SETUP.md**: Detailed installation instructions for conda, venv, and troubleshooting
- **validate.sh**: Automated validation script for local testing

#### CLI
- **Prediction tool** (`src/predict.py`):
  - Accept time-domain or frequency-domain CSV input
  - Output JSON with predicted label and probabilities
  - Save to `artifacts/prediction.json`

### Changed

#### Notebook
- **Refactored data loading**: Replaced inline code with calls to `src/data.safe_load()`
- **Simplified preprocessing**: Use functions from `src/preprocess` and `src/features`
- **Artifact export**: All figures now automatically export PNG alongside HTML
- **Headless compatibility**: Notebook runs successfully with papermill (no interactive prompts)

#### Dependencies
- **Updated requirements.txt**: Added version constraints and new packages:
  - `kaleido>=0.2.1` (PNG export)
  - `papermill>=2.5.0` (headless execution)
  - `pre-commit>=3.5.0` (hooks)
  - `ruff>=0.1.0` (linting)

### Fixed
- **Reproducibility**: All random operations now seeded consistently
- **Error handling**: Graceful fallbacks for network failures and missing packages
- **Code style**: Consistent formatting (black), import order (isort), linting (ruff)

### Removed
- **Old placeholder code**: Removed generic CSV download attempts from notebook
- **Inline duplication**: Extracted repeated preprocessing/feature code to modules
- `tests/test_basic.py`: Replaced with comprehensive test suite

## [1.0.0] - 2025-11-01 (Initial Demo)

### Added
- Basic Jupyter notebook with synthetic THz-like data
- Random Forest classifier
- Plotly interactive visualisations
- Simple CSV download with fallback

---

## Migration Notes (v1 → v2)

If you have an existing v1.0 notebook:

1. **Install new dependencies**: `pip install -r requirements.txt --upgrade`
2. **Update notebook imports**: Replace inline code with module imports (see updated notebook)
3. **Set `USE_DETRIS` flag**: Choose real or synthetic data
4. **Run validation**: `./validate.sh` to check setup

### Breaking Changes
- Data loading API changed (now uses `safe_load()` instead of inline functions)
- Feature extraction returns pandas DataFrame (was numpy array)
- Model artifacts now saved to `artifacts/` (was current directory)

### Deprecations
None in this release.

---

## Upcoming Features (Roadmap)

- [ ] THzDB integration (additional dataset source)
- [ ] Time-zero alignment preprocessing
- [ ] Reference normalisation (transmission/reflection)
- [ ] Cross-validation with hyperparameter tuning
- [ ] SVM and LightGBM alternative models
- [ ] Streamlit web interface
- [ ] Docker containerisation
- [ ] Extended feature set (cepstral coefficients, wavelet features)
