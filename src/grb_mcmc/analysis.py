"""Summaries are controlled by metadata present in files; no parameter of directory name assumptions."""

import json
from pathlib import Path
import emcee
import h5py
import numpy as np


def read_chain(path, config_path=None, discard=None):
    """Read new chains or legacy chains with explicit original config and discard.

    Legacy chains are read only. No guessed column order or implicit log transform.
    """
    with h5py.File(path, "r") as stream:
        if "parameter_names" in stream.attrs:
            names = json.loads(stream.attrs["parameter_names"])
            cfg = json.loads(stream.attrs["config_json"])
            burn = int(stream.attrs["burn_steps"])
        else:
            if config_path is None or discard is None:
                raise ValueError(
                    "Legacy chain needs its original config_path and explicit discard"
                )
            cfg = json.loads(Path(config_path).read_text())
            names = list(cfg["FREE_PARAMS"])
            burn = discard
    if discard is not None:
        burn = discard
    if type(burn) is not int or burn < 0:
        raise ValueError("discard must be a nonnegative integer")
    backend = emcee.backends.HDFBackend(str(path), read_only=True)
    if backend.shape[1] != len(names):
        raise ValueError("Parameter count does not match chain columns")
    if backend.iteration <= burn:
        raise ValueError("No production samples remain after discard")
    chain = backend.get_chain(discard=burn)
    probabilities = backend.get_log_prob(discard=burn, flat=True)
    if not np.isfinite(chain).all() or not np.isfinite(probabilities).all():
        raise ValueError("Chain contains nonfinite samples or probabilities")
    return names, cfg, chain, probabilities, backend


def posterior(path, config_path=None, discard=None):
    """Shared plotting/diagnostics payload with the same burn-in for MAP and samples."""
    names, cfg, chain, probs, _ = read_chain(path, config_path, discard)
    flat = chain.reshape(-1, len(names))
    return dict(
        names=names,
        config=cfg,
        samples=flat,
        best_params={
            **cfg.get("FIXED_PARAMS", {}),
            **dict(zip(names, flat[np.argmax(probs)])),
        },
        truth=cfg.get("TRUE_PARAMS", cfg.get("ALL_PARAMS", {})),
    )


def summarize(path):
    names, cfg, chain, probs, backend = read_chain(path)
    if not np.isfinite(chain).all() or not np.isfinite(probs).all():
        raise ValueError("Chain contains nonfinite samples or log probabilities")
    flat = chain.reshape(-1, chain.shape[-1])
    tau = None
    reliable = False
    try:
        estimate = emcee.autocorr.integrated_time(chain, quiet=False)
        if np.isfinite(estimate).all() and (estimate > 0).all():
            tau = estimate.tolist()
            reliable = bool(np.all(chain.shape[0] >= 50 * estimate))
    except emcee.autocorr.AutocorrError:
        pass
    q = np.percentile(flat, [16, 50, 84], axis=0)
    result = dict(
        fit_model=cfg["FIT_MODEL"],
        production_steps=chain.shape[0],
        autocorr_length_check_passed=reliable,
        autocorr_time=tau,
        estimated_effective_samples=(
            float(flat.shape[0] / max(tau)) if reliable else None
        ),
        mean_acceptance_fraction=float(np.mean(backend.accepted / backend.iteration)),
        max_log_posterior=float(probs.max()),
        parameters={
            n: dict(
                median=float(q[1, i]),
                errlo=float(q[1, i] - q[0, i]),
                errhi=float(q[2, i] - q[1, i]),
            )
            for i, n in enumerate(names)
        },
    )
    return result


def plot_chain(path, output):
    import matplotlib.pyplot as plt

    names, _, chain, _, _ = read_chain(path)
    fig, axes = plt.subplots(
        len(names), 1, squeeze=False, figsize=(9, 2.3 * len(names))
    )
    for index, name in enumerate(names):
        axes[index, 0].plot(chain[:, :, index], alpha=0.25, lw=0.5)
        axes[index, 0].set_ylabel(name)
    axes[-1, 0].set_xlabel("Production step")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def plot_corner(path, output):
    import matplotlib.pyplot as plt
    from .plotting import corner_plot

    item = posterior(path)
    fig, _ = corner_plot(
        {item["config"]["FIT_MODEL"]: item["samples"]},
        item["names"],
        truth=item["truth"],
    )
    fig.savefig(output, dpi=160)
    plt.close(fig)


def collect(base):
    """Include missing/failed jobs explicitly; use saved configs, never parse folder names."""
    from .config import load_config
    from .data import load_observations
    from .inference import residuals
    from .models import get_model

    rows = []
    for config in sorted(Path(base).glob("point_*/fit_*/config.json")):
        row = {"config": str(config)}
        try:
            cfg = load_config(config)
            path = cfg["SAMPLES_FILENAME"]
            result = summarize(path)
            row.update(
                {
                    k: v
                    for k, v in result.items()
                    if k not in ("parameters", "autocorr_time")
                }
            )
            for name, values in result["parameters"].items():
                row[f"rec_{name}"] = values["median"]
                row[f"rec_{name}_errlo"] = values["errlo"]
                row[f"rec_{name}_errhi"] = values["errhi"]
            for name, value in cfg.get("TRUE_PARAMS", {}).items():
                row[f"true_{name}"] = value
            names, _, chain, probs, _ = read_chain(path)
            best = chain.reshape(-1, len(names))[np.argmax(probs)]
            r, _ = residuals(
                best,
                get_model(cfg["FIT_MODEL"]),
                names,
                cfg["FIXED_PARAMS"],
                load_observations({**cfg, "GENERATE_DATA": False}),
                cfg["LIKELIHOOD"],
            )
            row["chi2_at_max_posterior"] = float(np.sum(r**2))
            row["status"] = "complete" if chain.shape[0] >= cfg["STEPS"] else "partial"
        except (OSError, ValueError, KeyError, AttributeError) as error:
            row.update(status="unavailable", error=str(error))
        rows.append(row)
    return rows
