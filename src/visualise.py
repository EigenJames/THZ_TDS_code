"""
Plotly figure builders with HTML and PNG export.

Functions to create interactive visualisations and export them as both
HTML (interactive) and PNG (static for documentation).
"""

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_time_domain(
    times: np.ndarray,
    X_time: np.ndarray,
    y_labels: np.ndarray,
    class_names: List[str],
    max_per_class: int = 10,
    output_dir: str = "artifacts",
    export_png: bool = True,
) -> go.Figure:
    """
    Create time-domain waveform plot with optional PNG export.

    Parameters
    ----------
    times : np.ndarray
        Time axis (seconds or same for all samples).
    X_time : np.ndarray (n_samples, n_points)
        Time-domain waveforms.
    y_labels : np.ndarray
        Class labels.
    class_names : List[str]
        List of class names.
    max_per_class : int
        Maximum samples to plot per class.
    output_dir : str
        Output directory for exports.
    export_png : bool
        Whether to export PNG.

    Returns
    -------
    go.Figure
        Plotly figure object.
    """
    fig = go.Figure()

    # Plot subset per class
    for cls in class_names:
        idx = np.where(y_labels == cls)[0][:max_per_class]
        for i in idx:
            t_axis = times[i] if times.ndim == 2 else times
            fig.add_trace(
                go.Scatter(
                    x=t_axis * 1e12,  # Convert to ps
                    y=X_time[i],
                    mode="lines",
                    name=cls,
                    opacity=0.6,
                    showlegend=False,
                )
            )

    # Add mean per class for legend
    for cls in class_names:
        idx = np.where(y_labels == cls)[0][:max_per_class]
        if len(idx) > 0:
            mean_sig = X_time[idx].mean(axis=0)
            t_axis = times[idx[0]] if times.ndim == 2 else times
            fig.add_trace(
                go.Scatter(
                    x=t_axis * 1e12,
                    y=mean_sig,
                    mode="lines",
                    name=f"{cls} (mean)",
                    line=dict(width=3),
                )
            )

    fig.update_layout(
        title="Time-domain waveforms (subset)",
        xaxis_title="Time (ps)",
        yaxis_title="Amplitude (a.u.)",
    )

    # Export
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path / "time_domain.html"))
    if export_png:
        try:
            fig.write_image(str(output_path / "time_domain.png"), width=1200, height=600)
        except Exception as e:
            print(f"Warning: Could not export PNG (kaleido required): {e}")

    return fig


def plot_spectra(
    freq_thz: np.ndarray,
    spectra: np.ndarray,
    y_labels: np.ndarray,
    class_names: List[str],
    max_per_class: int = 10,
    output_dir: str = "artifacts",
    export_png: bool = True,
) -> go.Figure:
    """
    Create frequency-domain spectra plot with optional PNG export.

    Parameters
    ----------
    freq_thz : np.ndarray
        Frequency axis (THz).
    spectra : np.ndarray (n_samples, n_freqs)
        Magnitude spectra.
    y_labels : np.ndarray
        Class labels.
    class_names : List[str]
        List of class names.
    max_per_class : int
        Maximum samples to plot per class.
    output_dir : str
        Output directory for exports.
    export_png : bool
        Whether to export PNG.

    Returns
    -------
    go.Figure
        Plotly figure object.
    """
    fig = go.Figure()

    # Plot subset per class
    for cls in class_names:
        idx = np.where(y_labels == cls)[0][:max_per_class]
        for i in idx:
            fig.add_trace(
                go.Scatter(
                    x=freq_thz,
                    y=spectra[i],
                    mode="lines",
                    name=cls,
                    opacity=0.6,
                    showlegend=False,
                )
            )

    # Add mean per class for legend
    for cls in class_names:
        idx = np.where(y_labels == cls)[0][:max_per_class]
        if len(idx) > 0:
            mean_spec = spectra[idx].mean(axis=0)
            fig.add_trace(
                go.Scatter(
                    x=freq_thz,
                    y=mean_spec,
                    mode="lines",
                    name=f"{cls} (mean)",
                    line=dict(width=3),
                )
            )

    fig.update_layout(
        title="Frequency-domain spectra (subset)",
        xaxis_title="Frequency (THz)",
        yaxis_title="|FFT| (a.u.)",
    )

    # Export
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path / "spectra.html"))
    if export_png:
        try:
            fig.write_image(str(output_path / "spectra.png"), width=1200, height=600)
        except Exception as e:
            print(f"Warning: Could not export PNG (kaleido required): {e}")

    return fig


def plot_pca_scatter(
    X_pca: np.ndarray,
    y: np.ndarray,
    output_dir: str = "artifacts",
    export_png: bool = True,
) -> go.Figure:
    """
    Create PCA scatter plot with optional PNG export.

    Parameters
    ----------
    X_pca : np.ndarray (n_samples, 2)
        PCA-transformed features.
    y : np.ndarray
        Class labels.
    output_dir : str
        Output directory for exports.
    export_png : bool
        Whether to export PNG.

    Returns
    -------
    go.Figure
        Plotly figure object.
    """
    df_pca = pd.DataFrame({"PC1": X_pca[:, 0], "PC2": X_pca[:, 1], "label": y})

    fig = px.scatter(
        df_pca,
        x="PC1",
        y="PC2",
        color="label",
        title="PCA (2D) of feature space",
    )

    # Export
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path / "pca_scatter.html"))
    if export_png:
        try:
            fig.write_image(str(output_path / "pca_scatter.png"), width=1000, height=600)
        except Exception as e:
            print(f"Warning: Could not export PNG (kaleido required): {e}")

    return fig


def plot_feature_importance(
    feature_names: List[str],
    importances: np.ndarray,
    output_dir: str = "artifacts",
    export_png: bool = True,
) -> go.Figure:
    """
    Create feature importance bar plot with optional PNG export.

    Parameters
    ----------
    feature_names : List[str]
        Feature names.
    importances : np.ndarray
        Feature importances from model.
    output_dir : str
        Output directory for exports.
    export_png : bool
        Whether to export PNG.

    Returns
    -------
    go.Figure
        Plotly figure object.
    """
    df_imp = pd.DataFrame({"feature": feature_names, "importance": importances})
    df_imp = df_imp.sort_values("importance", ascending=False)

    fig = px.bar(
        df_imp,
        x="feature",
        y="importance",
        title="Random Forest Feature Importances",
    )
    fig.update_layout(xaxis_tickangle=-45)

    # Export
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path / "feature_importances.html"))
    if export_png:
        try:
            fig.write_image(str(output_path / "feature_importances.png"), width=1200, height=600)
        except Exception as e:
            print(f"Warning: Could not export PNG (kaleido required): {e}")

    return fig