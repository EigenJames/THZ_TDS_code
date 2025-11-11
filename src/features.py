"""
Feature extraction: peaks, spectral moments, bandwidth proxy.

These functions extract compact feature vectors from frequency-domain spectra:
- Top-K spectral peaks (frequency and magnitude).
- Spectral moments (mean frequency, variance weighted by power).
- Bandwidth proxy (standard deviation of magnitude).
"""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


def extract_features(
    freq_thz: np.ndarray, mag: np.ndarray, topk: int = 5
) -> np.ndarray:
    """
    Extract feature vector from a single magnitude spectrum.

    Features:
    - Top-K peak frequencies (THz).
    - Top-K peak magnitudes.
    - Mean frequency (weighted by power).
    - Frequency variance (weighted by power).
    - Magnitude standard deviation.

    Parameters
    ----------
    freq_thz : np.ndarray
        Frequency axis in THz.
    mag : np.ndarray
        Magnitude spectrum.
    topk : int
        Number of top peaks to extract.

    Returns
    -------
    np.ndarray
        Feature vector of length (2*topk + 3).
    """
    # Find peaks
    peaks, _ = find_peaks(mag)

    # Select top-k peaks by magnitude
    if len(peaks) >= topk:
        top_idx = np.argsort(mag[peaks])[-topk:]
        sel_peaks = peaks[top_idx]
    else:
        sel_peaks = peaks

    # Extract peak features
    peak_freqs = freq_thz[sel_peaks] if len(sel_peaks) > 0 else np.array([])
    peak_mags = mag[sel_peaks] if len(sel_peaks) > 0 else np.array([])

    # Pad to topk
    if len(peak_freqs) < topk:
        pad = topk - len(peak_freqs)
        peak_freqs = np.pad(peak_freqs, (0, pad), mode="constant", constant_values=0)
        peak_mags = np.pad(peak_mags, (0, pad), mode="constant", constant_values=0)
    else:
        peak_freqs = peak_freqs[:topk]
        peak_mags = peak_mags[:topk]

    # Spectral moments
    power = mag**2
    power_sum = np.sum(power) + 1e-12
    mean_f = np.sum(freq_thz * power) / power_sum
    var_f = np.sum(((freq_thz - mean_f) ** 2) * power) / power_sum

    # Bandwidth proxy
    std_mag = np.std(mag)

    # Concatenate all features
    features = np.concatenate([peak_freqs, peak_mags, [mean_f, var_f, std_mag]])
    return features


def build_feature_matrix(
    freq_thz: np.ndarray, spectra: np.ndarray, topk: int = 5
) -> pd.DataFrame:
    """
    Build feature matrix from multiple spectra.

    Parameters
    ----------
    freq_thz : np.ndarray
        Frequency axis in THz.
    spectra : np.ndarray (n_samples, n_freqs)
        Magnitude spectra.
    topk : int
        Number of top peaks to extract per spectrum.

    Returns
    -------
    pd.DataFrame
        Feature matrix with columns: peak_f_1, peak_f_2, ..., peak_mag_1, ..., f_mean, f_var, mag_std.
    """
    feature_list = []
    for mag in spectra:
        feat = extract_features(freq_thz, mag, topk=topk)
        feature_list.append(feat)

    feature_array = np.array(feature_list)

    # Column names
    cols = (
        [f"peak_f_{i+1}" for i in range(topk)]
        + [f"peak_mag_{i+1}" for i in range(topk)]
        + ["f_mean", "f_var", "mag_std"]
    )

    return pd.DataFrame(feature_array, columns=cols)