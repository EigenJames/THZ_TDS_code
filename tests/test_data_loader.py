"""
Test data loading functionality using physics-driven synthetic data.
"""

import warnings

import numpy as np

from src.data import load_synthetic, safe_load


def test_synthetic_loader():
    """Test synthetic data generation."""
    times, X_time, y_labels, freq_thz, spectra = load_synthetic(
        n_samples_per_class=10, target_points=512, seed=42
    )

    # Check shapes
    assert times.shape == (512,)
    assert X_time.shape == (30, 512)  # 3 classes * 10 samples
    assert y_labels.shape == (30,)
    assert freq_thz.shape == (257,)  # 512//2 + 1
    assert spectra.shape == (30, 257)

    # Check label distribution
    unique_labels = set(y_labels.tolist())
    assert len(unique_labels) == 3
    assert "Polymer" in unique_labels
    assert "Ceramic" in unique_labels
    assert "Composite" in unique_labels

    # Check monotonicity of frequency axis
    assert np.all(np.diff(freq_thz) >= 0)

    # Check no NaN/Inf
    assert not np.any(np.isnan(X_time))
    assert not np.any(np.isinf(X_time))
    assert not np.any(np.isnan(spectra))
    assert not np.any(np.isinf(spectra))


def test_safe_load_synthetic():
    """Test safe_load with synthetic fallback."""
    times, X_time, y_labels, freq_thz, spectra, used_detris = safe_load(
        use_detris=False,
        n_samples_per_class=5,
        target_points=256,
        seed=42,
    )

    assert not used_detris
    assert X_time.shape == (15, 256)  # 3 classes * 5 samples
    assert freq_thz.shape == (129,)
    assert spectra.shape == (15, 129)


def test_safe_load_detris_flag_warns_and_returns_synthetic():
    """Requesting DETRIS should warn but still return synthetic data."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        times, X_time, y_labels, freq_thz, spectra, used_detris = safe_load(
            use_detris=True,
            materials=1,
            files_per_class=2,
            n_samples_per_class=5,
            target_points=256,
            seed=42,
        )

    assert any("synthetic" in str(w.message).lower() for w in caught)

    # Should receive synthetic data regardless of the flag
    assert not used_detris
    assert X_time.shape[0] == 15
    assert X_time.shape[1] == 256
    assert freq_thz.shape == (129,)

    # Check valid data
    assert not np.any(np.isnan(X_time))
    assert not np.any(np.isinf(X_time))
