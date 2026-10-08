"""JSON configuration with paths relative to the configuration file."""

import json
from pathlib import Path
import numpy as np

DEFAULTS = dict(
    FIXED_PARAMS={},
    BURN_STEPS=100,
    STEPS=1000,
    NWALKERS=32,
    SEED=42,
    WORKERS=1,
    PROGRESS=False,
    GENERATE_DATA=False,
    LIKELIHOOD="log_flux",
    FRACTIONAL_ERROR=0.1,
)


def load_config(path):
    path = Path(path).resolve()
    cfg = {**DEFAULTS, **json.loads(path.read_text())}
    for key in ("DATA_FILENAME", "SAMPLES_FILENAME"):
        cfg[key] = str((path.parent / cfg[key]).resolve())
    validate(cfg)
    return cfg


def validate(cfg):
    free = cfg["FREE_PARAMS"]
    names = list(free)
    from .models import MODEL_PARAMETERS

    supported = MODEL_PARAMETERS.get(cfg.get("FIT_MODEL"))
    if supported is not None and not set(names) <= supported:
        raise ValueError(
            f"{cfg['FIT_MODEL']} does not use these sampled parameters: {set(names) - supported}"
        )
    if not names or set(free) & set(cfg["FIXED_PARAMS"]):
        raise ValueError(
            "Free parameters must be nonempty and disjoint from fixed parameters"
        )
    if set(cfg["PRIORS"]) != set(free):
        raise ValueError("PRIORS must contain exactly the FREE_PARAMS keys")
    for name, value in free.items():
        bounds = np.asarray(cfg["PRIORS"][name], dtype=float)
        if (
            bounds.shape != (2,)
            or not np.isfinite(bounds).all()
            or not bounds[0] < bounds[1]
        ):
            raise ValueError(f"Invalid prior for {name}")
        if not np.isscalar(value) or not bounds[0] < value < bounds[1]:
            raise ValueError(
                f"Initial value of {name} must lie strictly inside its prior"
            )
    if not all(np.isscalar(v) and np.isfinite(v) for v in cfg["FIXED_PARAMS"].values()):
        raise ValueError("Fixed parameters must be finite numbers")
    for key, minimum in (
        ("NWALKERS", 2 * len(names)),
        ("BURN_STEPS", 0),
        ("STEPS", 1),
        ("WORKERS", 1),
        ("SEED", 0),
    ):
        if type(cfg[key]) is not int or cfg[key] < minimum:
            raise ValueError(f"{key} must be an integer >= {minimum}")
    if cfg["SEED"] >= 2**32:
        raise ValueError("SEED must be smaller than 2**32")
    if cfg.get("NDIM", len(names)) != len(names):
        raise ValueError(
            "NDIM does not match FREE_PARAMS; omit it to infer automatically"
        )
    if cfg["LIKELIHOOD"] not in ("log_flux", "linear_flux"):
        raise ValueError("LIKELIHOOD must be log_flux or linear_flux")
    if not np.isfinite(cfg["FRACTIONAL_ERROR"]) or cfg["FRACTIONAL_ERROR"] <= 0:
        raise ValueError("FRACTIONAL_ERROR must be positive")
    if Path(cfg["DATA_FILENAME"]).resolve() == Path(cfg["SAMPLES_FILENAME"]).resolve():
        raise ValueError("Data and chain paths must differ")
