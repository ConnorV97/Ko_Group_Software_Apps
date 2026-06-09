"""
dynes_fit.py

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
import nanonispy2 as ns2
import matplotlib.pyplot as plt

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
    return np.abs(np.real(z))


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

def ss_junction_dIdV(V, Delta, Gamma, T, V_ac, scale, offset, slope):
    """S-S junction dI/dV with tied gaps (tip and sample same material).

    Models a superconducting tip with gap Delta tunneling into a sample
    with the same gap. Coherence peaks appear at ±2Delta/e.
    """
    half_span = max(np.abs(V).max(), 8 * Delta) + max(10 * K_B * T, 3 * V_ac, 1e-4)
    E = np.linspace(-half_span, half_span, 2001)

    N = dynes_dos(E, Delta, Gamma)

    def fermi(E, T):
        x = np.clip(E / (K_B * T), -50, 50)
        return 1.0 / (1.0 + np.exp(x))

    f_E = fermi(E, T)

    # I(V) = ∫ N_tip(E') N_sample(E' + eV) [f(E') - f(E' + eV)] dE'
    I_of_V = np.zeros_like(E)
    for i, v in enumerate(E):
        N_shifted = np.interp(E + v, E, N, left=1.0, right=1.0)
        f_shifted = fermi(E + v, T)
        integrand = N * N_shifted * (f_E - f_shifted)
        I_of_V[i] = np.trapezoid(integrand, E)

    # dI/dV by differentiation
    dIdV_E = np.gradient(I_of_V, E)

    # Lock-in modulation convolution
    dE = E[1] - E[0]
    K_V = modulation_kernel(E, V_ac)
    dIdV_smeared = fftconvolve(dIdV_E, K_V, mode="same") * dE

    return scale * np.interp(V, E, dIdV_smeared) + offset + slope * V


def fit_ss(V, dIdV, T_kelvin, V_ac=2e-4, Delta_init=1.36e-3, Gamma_init=5e-5):
    """Fit an S-S junction model with tied gaps."""
    model = lmfit.Model(ss_junction_dIdV)
    dIdV = np.asarray(dIdV, dtype=float)
    scale_init = float(np.percentile(dIdV, 90) - np.percentile(dIdV, 10))
    offset_init = float(np.percentile(dIdV, 10))
    if scale_init <= 0:
        scale_init = 1.0

    params = model.make_params(
        Delta=dict(value=Delta_init, min=0.0, max=10e-3),
        Gamma=dict(value=Gamma_init, min=0.0, max=2e-3),
        T=dict(value=T_kelvin, vary=False),
        V_ac=dict(value=V_ac, vary=False),
        scale=dict(value=scale_init, min=0.0),
        offset=dict(value=offset_init),
        slope=dict(value=0.0),
    )
    return model.fit(dIdV, params, V=V)

def extract_data(file):

    dat = ns2.read.Spec(file)
    bias = dat.signals["Bias calc (V)"]
    didv = dat.signals["LI Demod 1 X (A)"]

    return bias, didv

# ---------------------------------------------------------------------------
# Synthetic-data helper for testing
# ---------------------------------------------------------------------------

def synthesize_spectrum(V, Delta, Gamma, T, V_ac, noise_frac=0.01, seed=None):
    """Build a synthetic dI/dV spectrum from known Dynes parameters."""
    rng = np.random.default_rng(seed)
    clean = dynes_dIdV(V, Delta, Gamma, T, V_ac, scale=1.0, offset=0.0, slope=0.0)
    return clean + noise_frac * rng.standard_normal(V.shape)


if __name__ == "__main__":
    file = r"C:\Users\cvernach\Desktop\20251217_Au(111)_4K_Auto_Temp\Au(111)_4k_Auto_Temp_0018.dat"

    bias, didv = extract_data(file)

    # Sort by ascending bias so the plot is left-to-right
    order = np.argsort(bias)
    bias = bias[order]
    didv = didv[order]

    # Normalize: divide by the median dI/dV in the wings (well above the gap)
    far_mask = np.abs(bias) > 0.6 * np.abs(bias).max()
    norm_val = np.median(didv[far_mask])
    didv_n = didv / norm_val if norm_val != 0 else didv

    # Fit (keyword args to avoid order confusion)
    T = 4.2
    V_ac = 0.2e-3

    # result = fit_dynes(
    #     bias, didv_n,
    #     T_kelvin=T,
    #     V_ac=V_ac,
    #     Delta_init=1.20e-3,
    #     Gamma_init=0.05e-3,
    #     fit_T= True
    # )

    result = fit_ss(
        bias, didv_n,
        T_kelvin=4.2,
        V_ac=0.2e-3,
        Delta_init=1.0e-3,  # one gap value; the peaks will appear at ±2Δ
        Gamma_init=0.05e-3,
    )
    print(result.fit_report())

    # Pull values out (in meV for the title)
    D = result.params["Delta"].value * 1e3
    Derr = (result.params["Delta"].stderr or 0) * 1e3
    G = result.params["Gamma"].value * 1e3
    Gerr = (result.params["Gamma"].stderr or 0) * 1e3

    print(result.fit_report())

    # Smooth model curve at higher resolution for a clean line
    V_dense = np.linspace(bias.min(), bias.max(), 1001)
    fit_dense = result.eval(V=V_dense)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(bias * 1e3, didv_n, "o", ms=4, color="C0", label="data (normalized)")
    ax.plot(V_dense * 1e3, fit_dense, "-", color="C1", lw=2, label="Dynes fit")
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("Bias (mV)")
    ax.set_ylabel("dI/dV (normalized)")
    ax.set_title(f"Δ = {D:.3f} ± {Derr:.3f} meV   Γ = {G:.4f} ± {Gerr:.4f} meV")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
