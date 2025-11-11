"""
Preprocessing utilities: baseline correction, smoothing, FFT.

These functions prepare raw THz time-domain waveforms for feature extraction:
- Baseline correction removes DC offset using a pre-pulse window.
- Savitzky-Golay smoothing reduces high-frequency noise.
- FFT converts time-domain signals to frequency-domain spectra.
"""

import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal import savgol_filter


def baseline_correct(x: np.ndarray, window_pre: int = 100) -> np.ndarray:
    """
    Remove baseline offset by subtracting the mean of the first window_pre points.

    Parameters
    ----------
    x : np.ndarray
        Input time-domain signal.
    window_pre : int
        Number of initial points to use for baseline estimation.

    Returns
    -------
    np.ndarray
        Baseline-corrected signal.
    """
    if len(x) < window_pre:
        window_pre = max(1, len(x) // 10)
    baseline = np.mean(x[:window_pre])
    return x - baseline


def savgol_smooth(
    x: np.ndarray, window: int = 21, poly: int = 3
) -> np.ndarray:
    """
    Apply Savitzky-Golay smoothing filter to reduce noise.

    Parameters
    ----------
    x : np.ndarray
        Input signal.
    window : int
        Window length (must be odd and <= len(x)).
    poly : int
        Polynomial order.

    Returns
    -------
    np.ndarray
        Smoothed signal.
    """
    if len(x) < 5:
        return x  # Too short to smooth

    # Ensure odd window length
    wl = min(window, len(x) - (1 - (len(x) % 2)))
    if wl % 2 == 0:
        wl -= 1
    if wl < 5:
        wl = 5

    # Ensure polynomial order is valid
    po = min(poly, wl - 2)
    if po < 1:
        po = 1

    return savgol_filter(x, window_length=wl, polyorder=po)


def compute_fft(
    y: np.ndarray, dt: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute FFT and return non-negative frequencies and magnitudes.

    Parameters
    ----------
    y : np.ndarray
        Time-domain signal.
    dt : float
        Time step (seconds).

    Returns
    -------
    freqs : np.ndarray
        Non-negative frequency axis (Hz).
    magnitudes : np.ndarray
        Magnitude spectrum.
    """
    Y = fft(y)
    freqs = fftfreq(len(y), d=dt)  # Hz
    # Keep only non-negative frequencies
    idx = freqs >= 0
    return freqs[idx], np.abs(Y[idx])


def preprocess_waveform(
    x: np.ndarray,
    window_pre: int = 100,
    smooth: bool = True,
    smooth_window: int = 21,
    smooth_poly: int = 3,
) -> np.ndarray:
    """
    Full preprocessing pipeline: baseline correction + optional smoothing.

    Parameters
    ----------
    x : np.ndarray
        Input time-domain signal.
    window_pre : int
        Baseline correction window size.
    smooth : bool
        Whether to apply smoothing.
    smooth_window : int
        Smoothing window size.
    smooth_poly : int
        Smoothing polynomial order.

    Returns
    -------
    np.ndarray
        Preprocessed signal.
    """
    y = baseline_correct(x, window_pre=window_pre)
    if smooth:
        y = savgol_smooth(y, window=smooth_window, poly=smooth_poly)
    return y