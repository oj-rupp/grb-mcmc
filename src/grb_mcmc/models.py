"""Importable model adapters; the core does not interpret physical parameters."""

from importlib import import_module
import numpy as np


PHYSICAL = {"p", "log_eps_e", "log_eps_B", "log_n_0", "log_E52", "z"}
MODEL_PARAMETERS = {
    "gs02_smooth": PHYSICAL | {"dL28"},
    "afterglowpy": PHYSICAL | {"dL28", "thetaCore", "thetaObs", "xi_N"},
    "VegasAfterglow": PHYSICAL | {"dL28", "thetaCore", "thetaObs", "Gamma0"},
    "PyFRS": PHYSICAL | {"thetaCore", "thetaObs", "Gamma0"},
}


def get_model(spec):
    """Resolve ``package.module:function`` (or the built-in ``toy`` model)."""
    if spec == "toy":
        return toy_flux
    aliases = {
        "gs02_smooth": "grb_mcmc.adapters.gs02:flux_density",
        "afterglowpy": "grb_mcmc.adapters.afterglow:flux_density",
        "VegasAfterglow": "grb_mcmc.adapters.vegas:flux_density",
        "PyFRS": "grb_mcmc.adapters.pyfrs:flux_density",
    }
    spec = aliases.get(spec, spec)
    module, sep, name = spec.partition(":")
    if not sep:
        raise ValueError(
            "Unknown model alias; use a built-in name or 'package.module:function'."
        )
    try:
        model = getattr(import_module(module), name)
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            f"Cannot load {spec}: missing {error.name}. Install the relevant optional model dependency."
        ) from error
    if not callable(model):
        raise TypeError(f"{spec} is not callable")
    return model


def evaluate(model, times, frequency, params):
    """Accept an array or legacy tuple ``(flux, *extras)`` for one band.

    Times are days, frequency is Hz, and flux density is mJy. Adapters are
    responsible for any parameter/unit conversion required by their engines.
    """
    result = model(np.asarray(times), np.array([frequency]), **params)
    if isinstance(result, tuple):
        result = result[0]
    flux = np.asarray(result, dtype=float).squeeze()
    flux = np.atleast_1d(flux)
    if flux.shape != np.asarray(times).shape:
        raise ValueError(f"Expected one flux per time; got {flux.shape}")
    return flux


def toy_flux(t_days, nu_hz, log_amplitude=0.0, alpha=1.2, beta=0.6):
    """Separable power law for software demos, NOT a physical afterglow model."""
    return (
        10**log_amplitude
        * np.asarray(t_days) ** (-alpha)
        * (nu_hz[0] / 1e14) ** (-beta)
    )
