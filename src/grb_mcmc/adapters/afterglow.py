"""Adapter for afterglowpy. See docs/models.md."""

import numpy as np
import afterglowpy as grb


def flux_density(t_arr_days, nu_arr, **kwargs):
    """Accepts the physical parameters along with two-dimensional arrays
    containing observer time, t, and observing frequency, nu, and
    returns the flux density Fnu (t, nu) at each time-frequency point."""
    # Assuming t in days

    p = kwargs.get("p")
    eps_e = 10 ** kwargs.get("log_eps_e")
    eps_B = 10 ** kwargs.get("log_eps_B")
    n_0 = 10 ** kwargs.get("log_n_0")
    E52 = 10 ** kwargs.get("log_E52")

    dL28 = kwargs.get("dL28")
    z = kwargs.get("z")
    xi_N = kwargs.get("xi_N")

    thetaCore = kwargs.get("thetaCore")
    thetaObs = kwargs.get("thetaObs")

    # For convenience, place arguments into a dict.
    Z = {
        "jetType": grb.jet.TopHat,  # Top-Hat jet
        "specType": grb.jet.SimpleSpec,  # Basic Synchrotron Spectrum
        "thetaObs": thetaObs,  # Viewing angle in radians
        "E0": E52 * 10**52,  # Isotropic-equivalent energy in erg
        "thetaCore": thetaCore,  # Half-opening angle in radians
        "n0": n_0,  # circumburst density in cm^{-3}
        "p": p,  # electron energy distribution index
        "epsilon_e": eps_e,  # epsilon_e
        "epsilon_B": eps_B,  # epsilon_B
        "xi_N": xi_N,  # Fraction of electrons accelerated
        "d_L": dL28 * 10**28,  # Luminosity distance in cm
        "z": z,
    }  # redshift

    # 1. Convert times to seconds
    t_arr_sec = np.array(t_arr_days) * grb.day2sec

    # 2. Use meshgrid to create 2D coordinate arrays of (times, frequencies)
    # T_grid and NU_grid will both have the shape: (len(t_arr_sec), len(nu_arr))
    T_grid, NU_grid = np.meshgrid(t_arr_sec, nu_arr, indexing="ij")

    # 3. Flatten them into matching 1D arrays
    t_flat = T_grid.flatten()
    nu_flat = NU_grid.flatten()

    # Evaluate the backend on paired time/frequency coordinates
    Fnu_flat = grb.fluxDensity(t_flat, nu_flat, **Z)

    # 5. Reshape the 1D output back to your desired 2D shape: (len(t_arr), len(nu_arr))
    Fnu = Fnu_flat.reshape(T_grid.shape)

    return Fnu
