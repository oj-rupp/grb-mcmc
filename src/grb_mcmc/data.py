"""Portable per-band NPZ observations, including actual observing times."""

from pathlib import Path
import numpy as np
from .models import evaluate, get_model


def get_bands(bands):
    result = {}
    for label, nu, start, stop, count in bands:
        if not isinstance(label, str) or not label or label in result:
            raise ValueError("Band labels must be nonempty and unique")
        if not np.isfinite([nu, start, stop]).all() or nu <= 0 or not 0 < start <= stop:
            raise ValueError(f"Invalid frequency or time interval for {label}")
        if type(count) is not int or count < 1:
            raise ValueError("Band point count must be a positive integer")
        result[label] = {"nu": nu, "t_arr": np.geomspace(start, stop, count)}
    if not result:
        raise ValueError("At least one band is required")
    return result


def validate_observations(obs, likelihood):
    if not obs:
        raise ValueError("No observations")
    for label, band in obs.items():
        t, flux, sigma = (np.asarray(band[k]) for k in ("t_arr", "Fnu", "sigma"))
        if t.ndim != 1 or not t.size or t.shape != flux.shape or t.shape != sigma.shape:
            raise ValueError(f"Mismatched/empty arrays in {label}")
        if not all(np.isfinite(a).all() for a in (t, flux, sigma)):
            raise ValueError(f"Nonfinite observations in {label}")
        if (
            (t <= 0).any()
            or (sigma <= 0).any()
            or not np.isfinite(band["nu"])
            or band["nu"] <= 0
        ):
            raise ValueError(
                f"Times, frequencies and uncertainties must be positive: {label}"
            )
        if likelihood == "log_flux" and (flux <= 0).any():
            raise ValueError(
                "log_flux requires positive observations; use linear_flux for signed fluxes"
            )


def save_observations(path, obs):
    """Write without overwriting existing data. Input fields: t_arr, nu, Fnu, sigma."""
    validate_observations(obs, "linear_flux")
    flat = {
        f"{key}_{label}": value
        for label, band in obs.items()
        for key, value in band.items()
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez(stream, **flat)


def generate_observations(cfg):
    model = get_model(cfg["GENERATOR_MODEL"])
    # Initial guesses and simulation truth are intentionally independent.
    truth = cfg.get("TRUE_PARAMS", cfg.get("ALL_PARAMS"))
    if truth is None:
        raise ValueError(
            "Synthetic generation requires explicit TRUE_PARAMS (or legacy ALL_PARAMS)"
        )
    params = {**cfg["FIXED_PARAMS"], **truth}
    rng = np.random.default_rng(cfg["SEED"])
    obs = get_bands(cfg["BANDS"])
    for band in obs.values():
        mean = evaluate(model, band["t_arr"], band["nu"], params)
        if not np.isfinite(mean).all() or (mean <= 0).any():
            raise ValueError("Synthetic model flux must be finite and positive")
        band["sigma"] = mean * cfg["FRACTIONAL_ERROR"]
        band["Fnu"] = mean + rng.normal(size=mean.size) * band["sigma"]
    validate_observations(obs, cfg["LIKELIHOOD"])
    save_observations(cfg["DATA_FILENAME"], obs)
    return obs


def load_observations(cfg):
    path = Path(cfg["DATA_FILENAME"])
    if not path.exists():
        if cfg["GENERATE_DATA"]:
            return generate_observations(cfg)
        raise FileNotFoundError(f"Observations not found: {path}")
    with np.load(path, allow_pickle=False) as flat:
        labels = [k[4:] for k in flat.files if k.startswith("Fnu_")]
        obs = {}
        legacy = None
        for label in labels:
            if f"t_arr_{label}" in flat and f"nu_{label}" in flat:
                grid = {
                    "t_arr": flat[f"t_arr_{label}"],
                    "nu": float(flat[f"nu_{label}"]),
                }
            else:
                if legacy is None:
                    legacy = get_bands(cfg["BANDS"])
                grid = legacy[label]
            obs[label] = {
                **grid,
                "Fnu": flat[f"Fnu_{label}"],
                "sigma": flat[f"sigma_{label}"],
            }
    validate_observations(obs, cfg["LIKELIHOOD"])
    return obs
