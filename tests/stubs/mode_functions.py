import numpy as np
def zeros_spectrum(t_roi, freq_min, freq_max, time, F, power_spectrum):
    """Zero bins outside the (time-varying) [freq_min, freq_max] envelope.
    Reconstructed from its call site; exact upstream behaviour may differ."""
    fmin = np.interp(time, t_roi, freq_min)
    fmax = np.interp(time, t_roi, freq_max)
    keep = (F[None, :] >= fmin[:, None]) & (F[None, :] <= fmax[:, None])
    return np.where(keep, power_spectrum, 0.0)
