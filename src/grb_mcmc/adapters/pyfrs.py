"""Adapter for PyFRS. See docs/models.md."""

import numpy as np
from PyFRS import fs


def flux_density(t_arr_days, nu_arr, **kwargs):
    day2sec = 86400

    p = kwargs.get("p")
    eps_e = 10 ** kwargs.get("log_eps_e")
    eps_B = 10 ** kwargs.get("log_eps_B")
    n_0 = 10 ** kwargs.get("log_n_0")
    E52 = 10 ** kwargs.get("log_E52")

    z = kwargs.get("z")
    theta_c = np.rad2deg(kwargs.get("thetaCore"))
    theta_obs = np.rad2deg(kwargs.get("thetaObs"))
    Gamma0 = kwargs.get("Gamma0")

    Z = {
        "z": z,
        "k": 0,
        "SSC": "No",
        "XIC": "No",
        "E0": E52 * 10**52,
        "Gamma0": Gamma0,
        "theta_j": theta_c,
        "theta_obs": theta_obs,
        "n18": n_0,
        "p": p,
        "epsilon_e": eps_e,
        "epsilon_B": eps_B,
    }

    t_arr_sec = np.array(t_arr_days) * day2sec
    nu_arr = np.array(nu_arr)
    Fnu = np.zeros((len(t_arr_sec), len(nu_arr)))

    for i, t in enumerate(t_arr_sec):
        for j, nu in enumerate(nu_arr):
            Fnu[i, j] = fs.FS_flux(np.array([t]), np.array([nu]), **Z)[0]

    return Fnu
