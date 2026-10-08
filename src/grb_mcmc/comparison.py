"""Config-based analysis for new and explicitly described legacy chains."""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from .analysis import posterior
from .data import load_observations
from .plotting import aligned_posteriors, corner_plot, light_curves, sed


def analysis_config(path):
    """Read only; resolve paths relative to config, then an exact local basename.

    The basename fallback supports copied legacy run directories. If both are
    present the explicit configured path wins. No fuzzy file/model guessing.
    """
    path = Path(path).resolve()
    cfg = json.loads(path.read_text())
    cfg.setdefault("LIKELIHOOD", "log_flux")
    cfg["GENERATE_DATA"] = False
    for key in ("DATA_FILENAME", "SAMPLES_FILENAME"):
        original = Path(cfg[key])
        resolved = original if original.is_absolute() else path.parent / original
        local = path.parent / original.name
        cfg[key] = str(resolved if resolved.exists() else local)
    return cfg


def compare_configs(
    paths, kind="corner", names=None, orientation="bottom_left", time=1.0, discard=None
):
    if kind not in ("corner", "lc", "sed"):
        raise ValueError("Unknown comparison kind")
    items = {}
    observations = None
    predictions = {}
    truth = None
    for path in paths:
        cfg = analysis_config(path)
        label = cfg["FIT_MODEL"]
        if label in items:
            raise ValueError(
                "Duplicate model labels; use Python plotting API for repeated models"
            )
        item = posterior(cfg["SAMPLES_FILENAME"], config_path=path, discard=discard)
        items[label] = item
        if truth is None:
            truth = item["truth"]
        elif truth != item["truth"]:
            raise ValueError("Configs have different injected truths; plot separately")
        if kind in ("lc", "sed"):
            current = load_observations(cfg)
            if observations is not None:
                same = list(current) == list(observations) and all(
                    np.array_equal(current[b][k], observations[b][k])
                    for b in current
                    for k in ("nu", "t_arr", "Fnu", "sigma")
                )
                if not same:
                    raise ValueError(
                        "Models were fitted to different observations; plot separately"
                    )
            observations = current
            predictions[label] = item["best_params"]
    if kind == "corner":
        data, names = aligned_posteriors(items, names)
        return corner_plot(data, names, truth=truth, orientation=orientation)[0]
    fig, ax = plt.subplots(figsize=(8, 5))
    if kind == "lc":
        light_curves(observations, predictions, ax=ax)
    else:
        sed(observations, predictions, time=time, ax=ax)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig
