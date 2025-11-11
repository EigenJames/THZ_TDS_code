# Quick Reference Card

## THz-TDS AI Classification Demo v2.0.0

### 🚀 First Time Setup

```bash
# Clone repo
git clone https://github.com/YOUR-USERNAME/thz-tds-ai-classification-demo.git
cd thz-tds-ai-classification-demo

# Setup environment (choose one)
conda env create -f environment.yml && conda activate thz-tds-ai  # Recommended
# or
python3 -m venv .venv && source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Setup pre-commit
pre-commit install
```

### ✅ Validate Installation

```bash
./validate.sh                 # Automated checks
pytest -v                     # Run tests
pre-commit run --all-files    # Check code quality
```

### 📓 Run Notebook

```bash
# Interactive
jupyter lab
# Then open notebooks/THz_TDS_AI_Classification_Demo.ipynb

# Headless (CI/testing)
papermill notebooks/THz_TDS_AI_Classification_Demo.ipynb artifacts/nb_executed.ipynb
```

### 🔮 CLI Prediction

```bash
# From time-domain waveform
python -m src.predict --input path/to/waveform.csv --domain time

# From frequency-domain spectrum  
python -m src.predict --input path/to/spectrum.csv --domain freq

# View output
cat artifacts/prediction.json
```

### 🧪 Testing

```bash
pytest -v                          # All tests
pytest tests/test_data_loader.py   # Data loading only
pytest -k "not online"             # Skip online tests (offline mode)
```

### 🎨 Code Quality

```bash
# Auto-format
black src/ tests/
isort src/ tests/

# Lint
ruff check src/ tests/ --fix

# Pre-commit (runs all)
pre-commit run --all-files
```

### 📦 Key Files

| File | Purpose |
|------|---------|
| `notebooks/THz_TDS_AI_Classification_Demo.ipynb` | Main interactive notebook |
| `src/data.py` | DETRIS loader + synthetic fallback |
| `src/preprocess.py` | Baseline, smoothing, FFT |
| `src/features.py` | Peak detection, spectral moments |
| `src/model.py` | Random Forest training |
| `src/predict.py` | CLI for inference |
| `artifacts/` | Generated outputs (HTML, PNG, models) |

### 🔧 Configuration

| Setting | Location | Default |
|---------|----------|---------|
| Use DETRIS data | Notebook cell 2 | `USE_DETRIS = True` |
| Materials | Notebook cell 2 | `MATERIALS = 3` |
| Files per class | Notebook cell 2 | `FILES_PER_CLASS = 24` |
| Random seed | Notebook cell 1 | `SEED = 42` |

### 📊 Outputs

After running notebook, `artifacts/` contains:
- `time_domain.html` / `.png` — Time-domain waveforms
- `spectra.html` / `.png` — Frequency spectra
- `pca_scatter.html` / `.png` — PCA visualisation
- `feature_importances.html` / `.png` — Feature importance
- `model.joblib`, `scaler.joblib`, `pca.joblib` — Trained models
- `prediction.json` — CLI prediction output

### ⚡ Quick Commands

```bash
# Full pipeline
pip install -r requirements.txt
pytest -v
papermill notebooks/THz_TDS_AI_Classification_Demo.ipynb artifacts/nb_executed.ipynb
ls -lah artifacts/

# Single prediction
python -m src.predict --input artifacts/sample_waveform.csv --domain time
```

### 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | Activate virtual environment |
| `externally-managed-environment` | Use conda or venv, not system Python |
| DETRIS download timeout | Set `USE_DETRIS = False` in notebook |
| PNG export fails | Install kaleido: `pip install kaleido --upgrade` |
| Tests fail offline | Normal—DETRIS test skipped, synthetic tests run |

### 📚 Documentation

- `README.md` — Comprehensive guide with pipeline diagram
- `SETUP.md` — Detailed installation instructions
- `CHANGELOG.md` — Version history and migration notes
- `COMPLETION_SUMMARY.md` — Project transformation summary

### 🎯 Data Sources

- **DETRIS**: Zenodo DOI [10.5281/zenodo.5079558](https://doi.org/10.5281/zenodo.5079558) (CC-BY 4.0)
- **Synthetic**: Generated on-the-fly (3 classes, deterministic)
- **Custom**: Edit `src/data.py` to add new loaders

### ⏱️ Typical Run Times (M1 Mac)

- DETRIS download: 1-2 min
- Synthetic generation: 10 sec
- Model training: 5-10 sec
- Total notebook: 3-4 min
- PNG export: 5-10 sec per figure

### 🔐 Reproducibility

All runs are deterministic:
- `PYTHONHASHSEED=0`
- `numpy.random.seed(42)`
- `random_state=42` in sklearn

Same inputs → same outputs (bit-for-bit).

---

**Version:** 2.0.0  
**Updated:** 10 November 2025  
**Licence:** MIT
