"""
Test feature extraction functionality.
"""

import numpy as np

from src.features import extract_features, build_feature_matrix


def test_extract_features_basic():
    """Test feature extraction from a single spectrum."""
    # Create synthetic spectrum
    freq_thz = np.linspace(0, 5, 512)
    mag = np.abs(np.sin(freq_thz * np.pi) + 0.5)

    features = extract_features(freq_thz, mag, topk=5)

    # Check length: 5 peak freqs + 5 peak mags + 3 moments = 13
    assert len(features) == 13

    # Check all finite
    assert np.all(np.isfinite(features))

    # Check non-negative peak magnitudes
    peak_mags = features[5:10]
    assert np.all(peak_mags >= 0)


def test_extract_features_fixed_length():
    """Test that feature vector has fixed length regardless of input size."""
    freq1 = np.linspace(0, 5, 256)
    mag1 = np.random.rand(256)

    freq2 = np.linspace(0, 5, 1024)
    mag2 = np.random.rand(1024)

    feat1 = extract_features(freq1, mag1, topk=5)
    feat2 = extract_features(freq2, mag2, topk=5)

    assert len(feat1) == len(feat2)
    assert len(feat1) == 13


def test_build_feature_matrix():
    """Test building feature matrix from multiple spectra."""
    freq_thz = np.linspace(0, 5, 512)
    n_samples = 20

    # Create multiple spectra
    spectra = np.random.rand(n_samples, 512)

    # Build feature matrix
    F = build_feature_matrix(freq_thz, spectra, topk=5)

    # Check shape
    assert F.shape == (n_samples, 13)

    # Check column names
    expected_cols = (
        [f"peak_f_{i+1}" for i in range(5)]
        + [f"peak_mag_{i+1}" for i in range(5)]
        + ["f_mean", "f_var", "mag_std"]
    )
    assert list(F.columns) == expected_cols

    # Check all finite
    assert not np.any(np.isnan(F.values))
    assert not np.any(np.isinf(F.values))


def test_extract_features_deterministic():
    """Test that feature extraction is deterministic."""
    freq_thz = np.linspace(0, 5, 512)
    mag = np.random.RandomState(42).rand(512)

    feat1 = extract_features(freq_thz, mag, topk=5)
    feat2 = extract_features(freq_thz, mag, topk=5)

    assert np.allclose(feat1, feat2)
