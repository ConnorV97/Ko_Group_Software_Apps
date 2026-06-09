"""
Dynes_fit.py

Dynes density-of-states fitting for STM tunneling spectroscopy.

Model:
    N(E)/N(0) = Re[ (E - i*Gamma) / sqrt((E - i*Gamma)^2 - Delta^2) ]

Observation model (what's actually measured as dI/dV at bias V):
    dI/dV(V) ∝ ∫ N(E) * K_thermal(E - eV, T) * K_modulation(E - eV, V_ac) dE

where K_thermal is the derivative of the Fermi-Dirac distribution and K_modulation
is the semicircular lock-in modulation kernel.

Usage:
    result = fit_dynes(V, dIdV, T_kelvin=4.2, V_ac=2e-4,
                      Delta_init=1.4e-3, Gamma_init=5e-5)
    print(result.fit_report())
    Delta = result.params['Delta'].value
    Delta_err = result.params['Delta'].stderr
"""

import numpy as np
from scipy.signal import fftconvolve
import lmfit

K_B = 8.617333262e-5  # eV/K


def dynes_dos(E, Delta, Gamma):
    """Dynes density of states normalized to the normal-state value N(0).

    Parameters
    ----------
    E : array
        Energy in eV (relative to the Fermi level).
    Delta : float
        Superconducting gap in eV.
    Gamma : float
        Dynes broadening parameter in eV.
    """
    z = (E - 1j * Gamma) / np.sqrt((E - 1j * Gamma) ** 2 - Delta ** 2 + 0j)
    return np.real(z)


def thermal_kernel(E, T):
    """-df/dE on a centered energy grid. Integral over E equals 1."""
    if T <= 0:
        k = np.zeros_like(E)
        k[np.argmin(np.abs(E))] = 1.0 / (E[1] - E[0])  # delta function
        return k
    x = E / (2 * K_B * T)
    x = np.clip(x, -50, 50)  # prevent cosh overflow far from center
    k = 1.0 / (4 * K_B * T * np.cosh(x) ** 2)
    return k / np.trapezoid(k, E)


def modulation_kernel(E, V_ac):
    """Lock-in modulation kernel (semicircular). Integral equals 1.

    V_ac is the RMS modulation amplitude in eV (same units as E).
    """
    if V_ac <= 0:
        k = np.zeros_like(E)
        k[np.argmin(np.abs(E))] = 1.0 / (E[1] - E[0])
        return k
    k = np.zeros_like(E)
    inside = np.abs(E) < V_ac
    k[inside] = (2.0 / (np.pi * V_ac)) * np.sqrt(1.0 - (E[inside] / V_ac) ** 2)
    integral = np.trapezoid(k, E)
    if integral > 0:
        k = k / integral
    return k


def dynes_dIdV(V, Delta, Gamma, T, V_ac, scale, offset, slope):
    """Dynes DOS convolved with thermal and lock-in kernels, evaluated at V.

    Parameters
    ----------
    V : array
        Bias voltages at which to evaluate (V or eV — be consistent throughout).
    Delta, Gamma : float
        Dynes parameters (same units as V).
    T : float
        Temperature in Kelvin (typically fixed to the measured value).
    V_ac : float
        Lock-in modulation amplitude (same units as V; typically fixed).
    scale : float
        Overall multiplicative scale on dI/dV.
    offset : float
        Additive constant baseline.
    slope : float
        Linear background slope (handles asymmetric tip DOS).
    """
    # Energy grid wide enough to cover the data range plus broadening tails
    half_span = max(np.abs(V).max(), 5 * Delta) + max(10 * K_B * T, 3 * V_ac, 1e-4)
    E = np.linspace(-half_span, half_span, 4001)
    dE = E[1] - E[0]

    N = dynes_dos(E, Delta, Gamma)
    K_T = thermal_kernel(E, T)
    K_V = modulation_kernel(E, V_ac)

    # Compose kernels first (cheaper than two convolutions of N)
    K = fftconvolve(K_T, K_V, mode="same") * dE
    K_int = np.trapezoid(K, E)
    if K_int > 0:
        K = K / K_int

    broadened = fftconvolve(N, K, mode="same") * dE

    # Interpolate model to experimental bias points and add background
    model = scale * np.interp(V, E, broadened) + offset + slope * V
    return model


def fit_dynes(
    V,
    dIdV,
    T_kelvin,
    V_ac=2e-4,
    Delta_init=1.4e-3,
    Gamma_init=5e-5,
    fit_T=False,
    fit_V_ac=False,
    weights=None,
):
    """Fit a measured spectrum to the Dynes model.

    Parameters
    ----------
    V : array
        Bias voltages (V), shape (N,).
    dIdV : array
        Measured differential conductance, shape (N,). Recommended to normalize
        so the normal-state value is ~1 before fitting.
    T_kelvin : float
        Measured temperature in Kelvin.
    V_ac : float
        Lock-in modulation amplitude (same units as V).
    Delta_init, Gamma_init : float
        Initial guesses for gap and broadening.
    fit_T, fit_V_ac : bool
        Whether to let T or V_ac vary. Typically False — fix to measured values.
    weights : array or None
        Per-point weights (1/sigma). If None, equal weights.

    Returns
    -------
    lmfit.model.ModelResult
        Use .params['Delta'].value and .params['Delta'].stderr to extract Delta
        and its uncertainty. .fit_report() prints everything.
    """
    model = lmfit.Model(dynes_dIdV)

    # Estimate scale and offset from data extremes for sensible initial guesses
    dIdV = np.asarray(dIdV, dtype=float)
    scale_init = np.percentile(dIdV, 90) - np.percentile(dIdV, 10)
    offset_init = np.percentile(dIdV, 10)
    if scale_init <= 0:
        scale_init = 1.0

    params = model.make_params(
        Delta=dict(value=Delta_init, min=0.0, max=20e-3),
        Gamma=dict(value=Gamma_init, min=0.0, max=5e-3),
        T=dict(value=T_kelvin, min=0.05, max=300.0, vary=fit_T),
        V_ac=dict(value=V_ac, min=0.0, max=10e-3, vary=fit_V_ac),
        scale=dict(value=scale_init, min=0.0),
        offset=dict(value=offset_init),
        slope=dict(value=0.0),
    )

    result = model.fit(dIdV, params, V=V, weights=weights)
    return result


# ---------------------------------------------------------------------------
# Synthetic-data helper for testing
# ---------------------------------------------------------------------------

def synthesize_spectrum(V, Delta, Gamma, T, V_ac, noise_frac=0.01, seed=None):
    """Build a synthetic dI/dV spectrum from known Dynes parameters."""
    rng = np.random.default_rng(seed)
    clean = dynes_dIdV(V, Delta, Gamma, T, V_ac, scale=1.0, offset=0.0, slope=0.0)
    return clean + noise_frac * rng.standard_normal(V.shape)


if __name__ == "__main__":
    # Self-test: synthesize a spectrum, fit it, check parameter recovery
    print("Self-test: Dynes fit on synthetic data")
    print("=" * 50)

    V_true = 0.001 * np.linspace(-5, 5, 401)  # ±5 mV in volts
    truth = dict(Delta=1.40e-3, Gamma=0.05e-3, T=4.2, V_ac=0.2e-3)

    for trial in range(3):
        dIdV = synthesize_spectrum(
            V_true, **truth, noise_frac=0.01, seed=trial
        )
        result = fit_dynes(
            V_true, dIdV,
            T_kelvin=truth["T"],
            V_ac=truth["V_ac"],
            Delta_init=1.0e-3,
            Gamma_init=1e-4,
        )
        D = result.params["Delta"].value * 1e3  # to meV
        D_err = result.params["Delta"].stderr * 1e3 if result.params["Delta"].stderr else float("nan")
        G = result.params["Gamma"].value * 1e3
        G_err = result.params["Gamma"].stderr * 1e3 if result.params["Gamma"].stderr else float("nan")
        print(f"trial {trial}: Delta = {D:.3f} ± {D_err:.3f} meV "
              f"(truth {truth['Delta']*1e3:.3f}) "
              f"|  Gamma = {G:.4f} ± {G_err:.4f} meV (truth {truth['Gamma']*1e3:.3f})")