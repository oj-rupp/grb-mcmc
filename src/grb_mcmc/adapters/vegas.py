"""Adapter for VegasAfterglow. See docs/models.md."""

from VegasAfterglow import ISM, TophatJet, Observer, Radiation, Model


def flux_density(t_arr_days, nu_arr, **kwargs):
    day2sec = 86400
    cgs_to_mJy = 1e26

    p = kwargs.get("p")
    eps_e = 10 ** kwargs.get("log_eps_e")
    eps_B = 10 ** kwargs.get("log_eps_B")
    n_0 = 10 ** kwargs.get("log_n_0")
    E52 = 10 ** kwargs.get("log_E52")

    dL28 = kwargs.get("dL28")
    z = kwargs.get("z")

    theta_c = kwargs.get("thetaCore")
    theta_obs = kwargs.get("thetaObs")
    Gamma0 = kwargs.get("Gamma0")

    t_arr = [t * day2sec for t in t_arr_days]

    model = Model(
        jet=TophatJet(theta_c=theta_c, E_iso=E52 * 10**52, Gamma0=Gamma0),
        medium=ISM(n_ism=n_0),
        observer=Observer(lumi_dist=dL28 * 10**28, z=z, theta_obs=theta_obs),
        fwd_rad=Radiation(eps_e=eps_e, eps_B=eps_B, p=p),
    )
    model_flux = model.flux_density_grid(t_arr, nu_arr).total

    # Return model_flux
    return model_flux * cgs_to_mJy
