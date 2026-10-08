"""
GS02-specific diagnostics, separate from the model-agnostic inference.
These checks are not always implemented into the full framework, as they would
need to further developed to ensure the GS02 model is fully integrated beyond the assumption
of Spectrum 1 from GS02.
"""

import numpy as np
from .adapters.gs02 import nu_a_b1, nu_m_b2, nu_c_b3


def transition_times(p, z, eps_e, eps_B, n_0, E52):
    """Original Spectrum 1 -> 2 estimated interval, in days."""
    scale = (
        (1 + z)
        * (eps_e * (p - 2) / (p - 1)) ** 2
        * eps_B**0.2
        * n_0 ** (-0.4)
        * E52**0.2
    )
    return 6.1e4 * scale, 1.2e6 * scale


def jet_time(theta_core, n_0, E52, z):
    """Original approximate observer-frame jet time (days; angle in radians)."""
    return (theta_core / 0.057) ** (8 / 3) * (1 + z) / 2 * (E52 / n_0 / 100) ** (1 / 3)


def deceleration_time(Gamma0, n_0, E52, z):
    """Original approximate observer-frame deceleration time (days)."""
    return 6.2 / 24 * (1 + z) * (E52 / n_0) ** (1 / 3) * Gamma0 ** (-8 / 3)


def band_diagnostics(bands, params):
    """Evaluate breaks at actual band times; coverage does not imply a constraint."""
    p, z = params["p"], params["z"]
    ee, eb, n, e = [
        10 ** params[k] for k in ("log_eps_e", "log_eps_B", "log_n_0", "log_E52")
    ]
    if p <= 2 or params.get("dL28", 1) <= 0:
        raise ValueError("GS02 diagnostics require p > 2 and positive distance")
    transition = transition_times(p, z, ee, eb, n, e)
    report = {"spectrum_1_to_2_days": list(transition), "bands": {}}
    for label, band in bands.items():
        times = np.asarray(band["t_arr"])
        a = nu_a_b1(p, z, ee, eb, n, e)
        m = nu_m_b2(p, z, ee, eb, n, e, times)
        c = nu_c_b3(p, z, ee, eb, n, e, times)
        report["bands"][label] = dict(
            nu_a_Hz=float(a),
            nu_m_range_Hz=[float(m.min()), float(m.max())],
            nu_c_range_Hz=[float(c.min()), float(c.max())],
            spectrum_1_ordering_at_all_times=bool(np.all((a < m) & (m < c))),
            below_nu_a=bool(band["nu"] < a),
            above_nu_c_at_any_time=bool(np.any(band["nu"] > c)),
            above_nu_c_at_all_times=bool(np.all(band["nu"] > c)),
            past_earliest_transition=bool(np.max(times) > transition[0]),
        )
    if "thetaCore" in params:
        report["jet_time_days"] = float(jet_time(params["thetaCore"], n, e, z))
    if "Gamma0" in params:
        report["deceleration_time_days"] = float(
            deceleration_time(params["Gamma0"], n, e, z)
        )
    return report
