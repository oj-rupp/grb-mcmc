"""Persistent ensemble sampling with explicit, validated continuation."""

from contextlib import nullcontext
import hashlib
from importlib.metadata import version
import platform
import json
import multiprocessing as mp
from pathlib import Path
import warnings
import emcee
import h5py
import numpy as np
from scipy.optimize import minimize
from .config import load_config
from .data import load_observations
from .inference import log_posterior
from .models import get_model


def fingerprint(cfg, obs):
    # Increasing the target length, moving files or changing worker count is OK.
    excluded = {
        "STEPS",
        "WORKERS",
        "PROGRESS",
        "SAMPLES_FILENAME",
        "DATA_FILENAME",
        "GENERATE_DATA",
    }
    payload = {k: v for k, v in cfg.items() if k not in excluded}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode())
    # Sorting JSON alone would conceal a change in the sampled column order.
    digest.update(json.dumps(list(cfg["FREE_PARAMS"])).encode())
    for label, band in obs.items():
        digest.update(label.encode())
        for key in ("t_arr", "nu", "Fnu", "sigma"):
            digest.update(np.asarray(band[key], dtype="<f8").tobytes())
    return digest.hexdigest()


def initialize_walkers(center, nwalkers, names, priors, log_prob, seed):
    rng = np.random.default_rng(seed)
    width = np.array([priors[n][1] - priors[n][0] for n in names])
    positions = []
    for _ in range(10000):
        candidate = center + 0.02 * width * rng.normal(size=len(names))
        if np.isfinite(log_prob(candidate)):
            positions.append(candidate)
            if len(positions) == nwalkers:
                return np.array(positions)
    raise ValueError(
        "Could not initialize finite walkers; check model, priors and starting values"
    )


def run(config_path, resume=False, production_steps=None):
    cfg = load_config(config_path)
    if production_steps is not None:
        if type(production_steps) is not int or production_steps < 1:
            raise ValueError("production_steps must be a positive integer")
        cfg["STEPS"] = production_steps
    path = Path(cfg["SAMPLES_FILENAME"])
    if path.exists() and not resume:
        raise FileExistsError(
            f"Chain exists: {path}. Use --resume to continue it or choose a new path."
        )
    if resume and not path.exists():
        raise FileNotFoundError(f"Cannot resume missing chain: {path}")
    obs = load_observations(cfg)
    model = get_model(cfg["FIT_MODEL"])
    names = list(cfg["FREE_PARAMS"])
    args = (model, names, cfg["FIXED_PARAMS"], obs, cfg["LIKELIHOOD"], cfg["PRIORS"])
    identity = fingerprint(cfg, obs)
    target = cfg["BURN_STEPS"] + cfg["STEPS"]
    if resume:
        with h5py.File(path, "r") as stream:
            if stream.attrs.get("config_fingerprint") != identity:
                raise ValueError(
                    "Resume refused: configuration, column order, or observations changed"
                )
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Reserve exclusively: another job must never reset our chain.
        with h5py.File(path, "x") as stream:
            stream.attrs["config_fingerprint"] = identity
            stream.attrs["config_json"] = json.dumps(cfg)
            stream.attrs["parameter_names"] = json.dumps(names)
            stream.attrs["burn_steps"] = cfg["BURN_STEPS"]
            stream.attrs["package_version"] = "0.2.0"
            stream.attrs["environment_json"] = json.dumps(
                {
                    "python": platform.python_version(),
                    **{
                        name: version(name)
                        for name in ("numpy", "scipy", "emcee", "h5py")
                    },
                }
            )
    backend = emcee.backends.HDFBackend(str(path))
    if not resume:
        backend.reset(cfg["NWALKERS"], len(names))
    remaining = target - backend.iteration
    if remaining <= 0:
        return path
    state = None
    if backend.iteration == 0:
        initial = np.array(list(cfg["FREE_PARAMS"].values()), dtype=float)

        def probability(theta):
            return log_posterior(theta, *args)

        if not np.isfinite(probability(initial)):
            raise ValueError("Initial parameters have nonfinite posterior")
        bounds = [
            (
                np.nextafter(cfg["PRIORS"][n][0], np.inf),
                np.nextafter(cfg["PRIORS"][n][1], -np.inf),
            )
            for n in names
        ]
        # Optimize the SAME likelihood used by the sampler.
        optimum = minimize(
            lambda theta: -probability(theta),
            initial,
            method="Nelder-Mead",
            bounds=bounds,
        )
        center = initial
        if optimum.success and np.isfinite(probability(optimum.x)):
            center = optimum.x
        else:
            warnings.warn(
                "Optimization failed; initializing around supplied starting values"
            )
        state = initialize_walkers(
            center, cfg["NWALKERS"], names, cfg["PRIORS"], probability, cfg["SEED"]
        )
    context = (
        mp.get_context("spawn").Pool(cfg["WORKERS"])
        if cfg["WORKERS"] > 1
        else nullcontext(None)
    )
    with context as pool:
        sampler = emcee.EnsembleSampler(
            cfg["NWALKERS"],
            len(names),
            log_posterior,
            args=args,
            pool=pool,
            backend=backend,
        )
        if backend.iteration == 0:
            sampler.random_state = np.random.RandomState(cfg["SEED"]).get_state()
        sampler.run_mcmc(state, remaining, progress=cfg["PROGRESS"])
    return path


def extend(config_path, steps=None, chunk=500, max_chunks=20, stability=0.01):
    """Append steps or run bounded chunks until length AND tau stability pass.

    This is a heuristic stopping rule, not proof of convergence. Calls the same
    validated resume path as ordinary fits; no independent sampler implementation.
    """
    cfg = load_config(config_path)
    path = Path(cfg["SAMPLES_FILENAME"])
    if not path.exists():
        raise FileNotFoundError(path)
    if steps is not None and (type(steps) is not int or steps < 1):
        raise ValueError("steps must be a positive integer")
    if (
        type(chunk) is not int
        or chunk < 1
        or type(max_chunks) is not int
        or max_chunks < 1
        or not 0 < stability < 1
    ):
        raise ValueError("Invalid extension limits")
    backend = emcee.backends.HDFBackend(str(path), read_only=True)
    if steps is not None:
        target = int(backend.iteration) + steps - cfg["BURN_STEPS"]
        if target < 1:
            raise ValueError("Extension must reach production; increase steps")
        run(config_path, resume=True, production_steps=target)
        return dict(stopped_by="requested_steps", total_steps=int(backend.iteration))
    old = None
    for _ in range(max_chunks):
        target = max(1, int(backend.iteration) + chunk - cfg["BURN_STEPS"])
        run(config_path, resume=True, production_steps=target)
        chain = backend.get_chain(discard=cfg["BURN_STEPS"])
        with np.errstate(invalid="ignore", divide="ignore"):
            tau = emcee.autocorr.integrated_time(chain, tol=0)
        valid = np.isfinite(tau).all() and (tau > 0).all()
        stable = valid and old is not None and np.all(abs(tau - old) / tau < stability)
        if stable and np.all(len(chain) >= 50 * tau):
            return dict(
                stopped_by="length_and_tau_stability",
                total_steps=int(backend.iteration),
                autocorr_time=tau.tolist(),
            )
        old = tau if valid else None
    return dict(stopped_by="max_chunks", total_steps=int(backend.iteration))
