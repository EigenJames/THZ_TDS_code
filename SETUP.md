# Setup Instructions

## Prerequisites

- Python 3.10 or later
- Git

## Installation

### Option 1: Conda (Recommended)

```bash
# Clone repository
git clone https://github.com/YOUR-USERNAME/thz-tds-ai-classification-demo.git
cd thz-tds-ai-classification-demo

# Create conda environment from environment.yml
conda env create -f environment.yml
conda activate thz-tds-ai

# Install additional pip-only packages
pip install kaleido ruff pre-commit papermill
```

### Option 2: Virtual Environment (venv)

```bash
# Clone repository
git clone https://github.com/YOUR-USERNAME/thz-tds-ai-classification-demo.git
cd thz-tds-ai-classification-demo

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Option 3: System Python (Not Recommended for macOS 15+)

If you have externally-managed Python (macOS 15+, some Linux distros), you **must** use a virtual environment (Option 1 or 2) or install packages with `--break-system-packages` flag (not recommended).

## Verify Installation

Run the validation script:

```bash
./validate.sh
```

Or manually check:

```bash
python3 -c "import numpy, pandas, scipy, sklearn, plotly; print('✓ Core packages OK')"
pytest --version
black --version
```

## Running the Notebook

### Jupyter Lab

```bash
jupyter lab
# Open notebooks/THz_TDS_AI_Classification_Demo.ipynb
```

### VS Code

1. Open the project folder in VS Code
2. Install "Jupyter" extension
3. Open `notebooks/THz_TDS_AI_Classification_Demo.ipynb`
4. Select Python interpreter (`.venv` or `thz-tds-ai` conda env)
5. Run cells

### Headless (CI/Testing)

```bash
papermill notebooks/THz_TDS_AI_Classification_Demo.ipynb artifacts/nb_executed.ipynb
```

## Development Setup

### Pre-commit Hooks

```bash
pre-commit install
pre-commit run --all-files  # Check all files
```

Hooks will run automatically on `git commit`:
- Trailing whitespace removal
- End-of-file fixer
- Black (formatting)
- isort (import sorting)
- Ruff (linting)

### Code Quality

```bash
# Auto-format
black src/ tests/
isort src/ tests/

# Lint
ruff check src/ tests/ --fix

# Type checking (optional)
mypy src/
```

## Testing

### Run All Tests

```bash
pytest -v
```

### Run Specific Test Files

```bash
pytest tests/test_data_loader.py -v
pytest tests/test_features.py -v
pytest tests/test_model.py -v
```

### Skip Online Tests (Offline Mode)

```bash
pytest -v -k "not online"
```

## Troubleshooting

### Import Errors

If you see `ModuleNotFoundError`, ensure:
1. Virtual environment is activated
2. All packages installed: `pip install -r requirements.txt`
3. `src/` is in Python path (notebook does `sys.path.insert(0, "..")`)

### Kaleido PNG Export Issues

If PNG export fails in the notebook:
```bash
pip install kaleido --upgrade
```

On some systems, you may need to install system dependencies for kaleido.

### DETRIS Download Timeout

If DETRIS download times out:
1. Set `USE_DETRIS = False` in notebook to use synthetic data
2. Or increase timeout in `src/data.py` (`timeout=300`)

### Externally-Managed Python (macOS 15+)

Error: `error: externally-managed-environment`

**Solution:** Use Option 1 (conda) or Option 2 (venv). Do **not** install directly to system Python.

## Environment Variables

Set these for reproducibility:

```bash
export PYTHONHASHSEED=0  # Deterministic hashing
export OMP_NUM_THREADS=1 # Reproducible numpy/scipy (optional)
```

## Next Steps

1. Run validation: `./validate.sh`
2. Open notebook: `jupyter lab` or VS Code
3. Run all cells to generate artifacts
4. Check `artifacts/` for outputs (HTML, PNG, model files)
