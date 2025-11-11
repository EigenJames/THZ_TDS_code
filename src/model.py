"""
Model helpers: train/evaluate, save/load pipeline.

Functions for training Random Forest classifier and saving/loading model artifacts.
"""

import joblib
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def train_random_forest(
    X: np.ndarray,
    y: np.ndarray,
    n_estimators: int = 300,
    random_state: int = 42,
    class_weight: str = "balanced",
) -> RandomForestClassifier:
    """
    Train a Random Forest classifier.

    Parameters
    ----------
    X : np.ndarray (n_samples, n_features)
        Feature matrix.
    y : np.ndarray (n_samples,)
        Target labels.
    n_estimators : int
        Number of trees.
    random_state : int
        Random seed.
    class_weight : str
        Class weight strategy.

    Returns
    -------
    RandomForestClassifier
        Trained classifier.
    """
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        class_weight=class_weight,
    )
    clf.fit(X, y)
    return clf


def save_artifacts(
    model: RandomForestClassifier,
    scaler: StandardScaler,
    pca: PCA,
    output_dir: str = "artifacts",
) -> None:
    """
    Save model, scaler, and PCA to disk.

    Parameters
    ----------
    model : RandomForestClassifier
        Trained model.
    scaler : StandardScaler
        Fitted scaler.
    pca : PCA
        Fitted PCA transformer.
    output_dir : str
        Output directory path.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, output_path / "model.joblib")
    joblib.dump(scaler, output_path / "scaler.joblib")
    joblib.dump(pca, output_path / "pca.joblib")

    print(f"Saved artifacts to {output_path}/")


def load_artifacts(
    input_dir: str = "artifacts",
) -> tuple[RandomForestClassifier, StandardScaler, PCA]:
    """
    Load model, scaler, and PCA from disk.

    Parameters
    ----------
    input_dir : str
        Input directory path.

    Returns
    -------
    model : RandomForestClassifier
        Loaded model.
    scaler : StandardScaler
        Loaded scaler.
    pca : PCA
        Loaded PCA transformer.
    """
    input_path = Path(input_dir)

    model = joblib.load(input_path / "model.joblib")
    scaler = joblib.load(input_path / "scaler.joblib")
    pca = joblib.load(input_path / "pca.joblib")

    print(f"Loaded artifacts from {input_path}/")
    return model, scaler, pca