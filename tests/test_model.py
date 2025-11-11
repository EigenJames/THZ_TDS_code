"""
Test model training and evaluation.
"""

import numpy as np
from sklearn.model_selection import train_test_split

from src.data import load_synthetic
from src.features import build_feature_matrix
from src.model import train_random_forest


def test_train_model_small():
    """Test training on small synthetic dataset."""
    # Generate small synthetic dataset
    times, X_time, y_labels, freq_thz, spectra = load_synthetic(
        n_samples_per_class=30, target_points=512, seed=42
    )

    # Extract features
    F = build_feature_matrix(freq_thz, spectra, topk=5)
    X = F.values
    y = y_labels

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Train model
    clf = train_random_forest(X_train, y_train, n_estimators=50, random_state=42)

    # Evaluate
    accuracy = clf.score(X_test, y_test)

    # Should achieve reasonable accuracy on synthetic data
    assert accuracy > 0.7, f"Accuracy too low: {accuracy:.3f}"

    # Check predictions
    y_pred = clf.predict(X_test)
    assert len(y_pred) == len(y_test)
    assert set(y_pred).issubset(set(y))


def test_model_reproducibility():
    """Test that model training is reproducible with same seed."""
    times, X_time, y_labels, freq_thz, spectra = load_synthetic(
        n_samples_per_class=20, target_points=256, seed=42
    )

    F = build_feature_matrix(freq_thz, spectra, topk=5)
    X = F.values
    y = y_labels

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Train two models with same seed
    clf1 = train_random_forest(X_train, y_train, n_estimators=50, random_state=42)
    clf2 = train_random_forest(X_train, y_train, n_estimators=50, random_state=42)

    # Predictions should be identical
    y_pred1 = clf1.predict(X_test)
    y_pred2 = clf2.predict(X_test)

    assert np.array_equal(y_pred1, y_pred2)


def test_model_classes():
    """Test that model learns all classes."""
    times, X_time, y_labels, freq_thz, spectra = load_synthetic(
        n_samples_per_class=30, target_points=512, seed=42
    )

    F = build_feature_matrix(freq_thz, spectra, topk=5)
    X = F.values
    y = y_labels

    clf = train_random_forest(X, y, n_estimators=50, random_state=42)

    # Check that model knows all classes
    expected_classes = sorted(set(y.tolist()))
    assert list(clf.classes_) == expected_classes

    # Check feature importances exist
    assert len(clf.feature_importances_) == X.shape[1]
    assert np.all(clf.feature_importances_ >= 0)
    assert np.isclose(clf.feature_importances_.sum(), 1.0)
