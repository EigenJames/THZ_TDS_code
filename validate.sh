#!/bin/bash
# Validation script for THz-TDS AI Classification Demo
# Run this after installing dependencies

set -e

echo "==================================="
echo "THz-TDS AI Classification Demo"
echo "Validation Script"
echo "==================================="
echo

# Check Python version
echo "✓ Checking Python version..."
python3 --version

# Create artifacts directory
echo "✓ Creating artifacts directory..."
mkdir -p artifacts

# Run linting (if tools available)
echo
echo "==================================="
echo "Code Quality Checks"
echo "==================================="

if command -v black &> /dev/null; then
    echo "✓ Running black (formatting check)..."
    black --check src/ tests/ || echo "⚠ Black formatting issues found (run 'black src/ tests/' to fix)"
else
    echo "⚠ black not found - skip formatting check"
fi

if command -v isort &> /dev/null; then
    echo "✓ Running isort (import sorting check)..."
    isort --check-only src/ tests/ || echo "⚠ Import sorting issues found (run 'isort src/ tests/' to fix)"
else
    echo "⚠ isort not found - skip import sorting check"
fi

if command -v ruff &> /dev/null; then
    echo "✓ Running ruff (linting)..."
    ruff check src/ tests/ || echo "⚠ Linting issues found"
else
    echo "⚠ ruff not found - skip linting"
fi

# Run tests
echo
echo "==================================="
echo "Test Suite"
echo "==================================="

if command -v pytest &> /dev/null; then
    echo "✓ Running pytest..."
    PYTHONHASHSEED=0 pytest -v || echo "⚠ Some tests failed"
else
    echo "⚠ pytest not found - skip tests"
    echo "  To run tests: python3 -m pytest -v"
fi

# Run notebook (if papermill available)
echo
echo "==================================="
echo "Notebook Execution"
echo "==================================="

if command -v papermill &> /dev/null; then
    echo "✓ Executing notebook with papermill..."
    PYTHONHASHSEED=0 papermill notebooks/THz_TDS_AI_Classification_Demo.ipynb artifacts/nb_executed.ipynb || echo "⚠ Notebook execution failed"
else
    echo "⚠ papermill not found - skip notebook execution"
    echo "  To run notebook: papermill notebooks/THz_TDS_AI_Classification_Demo.ipynb artifacts/nb_executed.ipynb"
fi

# List artifacts
echo
echo "==================================="
echo "Generated Artifacts"
echo "==================================="
ls -lah artifacts/ || echo "No artifacts generated yet"

echo
echo "==================================="
echo "Validation Complete!"
echo "==================================="
echo
echo "Next steps:"
echo "1. If linting issues found: run 'black src/ tests/' and 'isort src/ tests/'"
echo "2. Install pre-commit hooks: 'pre-commit install' then 'pre-commit run --all-files'"
echo "3. Open notebook in Jupyter Lab or VS Code and run interactively"
echo
