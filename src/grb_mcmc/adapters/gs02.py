"""GS02 Spectrum 1 ISM implementation.

Assumes nu_a < nu_m < nu_c. See grb_mcmc.physics for diagnostics.
"""

import numpy as np


def flux_density(t_arr, nu_arr, **kwargs):
    """Accepts the physical parameters along with two-dimensional arrays
    containing observer time, t, and observing frequency, nu, and
    returns the flux density Fnu (t, nu) at each time-frequency point.
    Includes the smoothing from Equations 1, 4, and 5 from GS02."""
    # Assuming t in days
    # Assuming Spectrum 1

    p = kwargs.get("p")
    z = kwargs.get("z")
    eps_e = 10 ** kwargs.get("log_eps_e")
    eps_B = 10 ** kwargs.get("log_eps_B")
    n_0 = 10 ** kwargs.get("log_n_0")
    E52 = 10 ** kwargs.get("log_E52")
    dL28 = kwargs.get("dL28")

    Fnu = np.zeros(((len(t_arr)), len(nu_arr)))
    nu_a_arr = np.zeros(len(t_arr))
    nu_m_arr = np.zeros(len(t_arr))
    nu_c_arr = np.zeros(len(t_arr))

    for t_i, t in enumerate(t_arr):
        nu_a = nu_a_b1(p, z, eps_e, eps_B, n_0, E52)
        nu_1 = nu_a
        nu_a_arr[t_i] = nu_a

        nu_m = nu_m_b2(p, z, eps_e, eps_B, n_0, E52, t)
        nu_2 = nu_m
        nu_m_arr[t_i] = nu_m

        nu_c = nu_c_b3(p, z, eps_e, eps_B, n_0, E52, t)
        nu_3 = nu_c
        nu_c_arr[t_i] = nu_c

        Fnu_b1 = Fnu_ext_b1(p, z, eps_e, eps_B, n_0, E52, dL28, t)

        for nu_i, nu in enumerate(nu_arr):
            Fnu[t_i, nu_i] = Fnu_1_Eq5(Fnu_b1, nu, nu_1, nu_2, nu_3, p)

    return Fnu, nu_a_arr, nu_m_arr, nu_c_arr


def nu_a_b1(p, z, eps_e, eps_B, n0, E52):
    p_bar = (p - 2) / (p - 1)

    gam_1 = (
        1.24 * (p - 1) ** 0.6 / (3 * p + 2) ** 0.6 * 10**9 * (1 + z) ** -1 * p_bar**-1
    )
    nu_a = gam_1 * eps_e**-1 * eps_B**0.2 * n0**0.6 * E52**0.2
    return nu_a


def nu_m_b2(p, z, eps_e, eps_B, n0, E52, t_days):
    p_bar = (p - 2) / (p - 1)

    gam_2 = 3.73 * (p - 0.67) * 10**15 * (1 + z) ** 0.5 * p_bar**2 * t_days**-1.5
    nu_m = gam_2 * eps_e**2 * eps_B**0.5 * n0**0 * E52**0.5
    return nu_m


def nu_c_b3(p, z, eps_e, eps_B, n0, E52, t_days):

    gam_3 = (
        6.37 * (p - 0.46) * 10**13 * np.exp(-1.16 * p) * (1 + z) ** -0.5 * t_days**-0.5
    )
    nu_c = gam_3 * eps_e**0 * eps_B**-1.5 * n0**-1 * E52**-0.5
    return nu_c


def Fnu_ext_b1(p, z, eps_e, eps_B, n0, E52, dL28, t_days):
    p_bar = (p - 2) / (p - 1)
    gam = (
        0.647
        * (p - 1) ** (6 / 5)
        / (3 * p - 1)
        / (3 * p + 2) ** (1 / 5)
        / p_bar
        * (1 + z) ** (1 / 2)
        * t_days ** (1 / 2)
        * dL28 ** (-2)
    )
    return gam / eps_e * eps_B ** (2 / 5) * n0 ** (7 / 10) * E52 ** (9 / 10)


def Fnu_ext_b2(p, z, eps_e, eps_B, n0, E52, dL28):
    return 9.93 * (p + 0.14) * (1 + z) * dL28**-2 * eps_B**0.5 * n0**0.5 * E52


def Fnu_tab1_G(p, z, eps_e, eps_B, n0, E52, dL28, t_days, nu):
    p_bar = (p - 2) / (p - 1)
    nu_14 = nu / 1e14

    gam_G = (
        0.461
        * (p - 0.04)
        * np.exp(2.53 * p)
        * (1 + z) ** ((3 + p) / 4)
        * p_bar ** (p - 1)
        * dL28 ** (-2)
        * nu_14 ** ((1 - p) / 2)
        * t_days ** (3 * (1 - p) / 4)
    )
    coeff_G = [p - 1, (1 + p) / 4, 0.5, (3 + p) / 4]

    Fnu_m = (
        gam_G
        * eps_e ** coeff_G[0]
        * eps_B ** coeff_G[1]
        * n0 ** coeff_G[2]
        * E52 ** coeff_G[3]
    )
    return Fnu_m


def Fnu_tab1_D(p, z, eps_e, eps_B, n0, E52, dL28, t_days, nu):
    p_bar = (p - 2) / (p - 1)
    nu_14 = nu / 1e14

    gam_D = (
        27.9
        * (p - 1)
        / (3 * p - 1)
        * (1 + z) ** (5 / 6)
        * p_bar ** (-2 / 3)
        * t_days**0.5
        * dL28**-2
        * nu_14 ** (1 / 3)
    )
    coeff_D = [-2 / 3, 1 / 3, 0.5, 5 / 6]

    Fnu_m = (
        gam_D
        * eps_e ** coeff_D[0]
        * eps_B ** coeff_D[1]
        * n0 ** coeff_D[2]
        * E52 ** coeff_D[3]
    )
    return Fnu_m


def Fnu_Eq1(Fnu_ext, nu, nu_b, s, beta1, beta2):
    return Fnu_ext * ((nu / nu_b) ** (-s * beta1) + (nu / nu_b) ** (-s * beta2)) ** (
        -1 / s
    )


def tilF_b_Eq4(nu, nu_b, s, beta1, beta2):
    return (1 + (nu / nu_b) ** (s * (beta1 - beta2))) ** (-1 / s)


def Fnu_1_Eq5(Fnu_ext_b1, nu, nu_1, nu_2, nu_3, p):
    # Fnu = Fnu_1 * tilFnu_2 * tilFnu3, see Equations 1, 4, 5 GS02
    Fnu = (
        Fnu_Eq1(Fnu_ext_b1, nu, nu_1, 1.64, 2, 1 / 3)
        * tilF_b_Eq4(nu, nu_2, (1.84 - 0.40 * p), 1 / 3, (1 - p) / 2)
        * tilF_b_Eq4(nu, nu_3, (1.15 - 0.06 * p), (1 - p) / 2, -p / 2)
    )
    return Fnu
