"""One sweep-layout implementation for histograms, contours, LCs and SEDs."""

from pathlib import Path
import matplotlib.pyplot as plt
from .analysis import posterior, summarize
from .config import load_config
from .data import load_observations
from .plotting import density_contours, color_for, light_curves, sed


def sweep_grid(
    base, x, y, kind="hist", parameters=None, models=None, time=1.0, truth_models=False
):
    if kind not in ("hist", "contour", "lc", "sed"):
        raise ValueError("Unknown grid kind")
    count = {"hist": 1, "contour": 2, "lc": 0, "sed": 0}[kind]
    parameters = parameters or []
    if len(parameters) != count:
        raise ValueError(f"{kind} needs {count} parameter names")
    cells = {}
    for path in sorted(Path(base).glob("point_*/fit_*/config.json")):
        cfg = load_config(path)
        if models and cfg["FIT_MODEL"] not in models:
            continue
        truth = cfg["TRUE_PARAMS"]
        key = (truth[x], truth[y])
        cell = cells.setdefault(key, [])
        # Do not silently merge points differing in an unplotted third axis.
        if cell and cell[0][0].parent.parent != path.parent.parent:
            raise ValueError(
                "Multiple sweep points share these axes; select a two-dimensional sweep"
            )
        cell.append((path, cfg))
    if not cells:
        raise ValueError("No matching sweep configs")
    xs = sorted({k[0] for k in cells})
    ys = sorted({k[1] for k in cells}, reverse=True)
    fig, axes = plt.subplots(
        len(ys), len(xs), squeeze=False, figsize=(4 * len(xs), 3.2 * len(ys))
    )
    all_models = list(
        dict.fromkeys(
            cfg["FIT_MODEL"] for entries in cells.values() for _, cfg in entries
        )
    )
    for row, yv in enumerate(ys):
        for col, xv in enumerate(xs):
            ax = axes[row, col]
            entries = cells.get((xv, yv), [])
            ax.set_title(f"{x}={xv:g}, {y}={yv:g}", fontsize=9)
            if not entries:
                ax.text(0.5, 0.5, "Missing point", ha="center", transform=ax.transAxes)
                continue
            predictions = {}
            problems = []
            obs = None
            for path, cfg in entries:
                model = cfg["FIT_MODEL"]
                color = color_for(model, all_models.index(model))
                try:
                    if truth_models:
                        predictions[model] = {
                            **cfg["FIXED_PARAMS"],
                            **cfg["TRUE_PARAMS"],
                        }
                        if kind in ("hist", "contour"):
                            raise ValueError("truth_models is only for LC/SED grids")
                    else:
                        item = posterior(cfg["SAMPLES_FILENAME"])
                        predictions[model] = item["best_params"]
                        samples = item["samples"]
                        names = item["names"]
                        if not summarize(cfg["SAMPLES_FILENAME"])[
                            "autocorr_length_check_passed"
                        ]:
                            problems.append(f"{model}: short/uncertain chain")
                        if kind == "hist":
                            values = samples[:, names.index(parameters[0])]
                            ax.hist(
                                values,
                                bins=25,
                                density=True,
                                histtype="step",
                                color=color,
                                label=model,
                            )
                            ax.axvline(
                                cfg["TRUE_PARAMS"][parameters[0]],
                                color="black",
                                ls="--",
                                lw=1,
                            )
                            ax.set(xlabel=parameters[0], ylabel="Density")
                        elif kind == "contour":
                            a, b = parameters
                            density_contours(
                                ax,
                                samples[:, names.index(a)],
                                samples[:, names.index(b)],
                                color,
                            )
                            ax.plot(cfg["TRUE_PARAMS"][a], cfg["TRUE_PARAMS"][b], "k*")
                            ax.set(xlabel=a, ylabel=b)
                    if kind in ("lc", "sed") and obs is None:
                        obs = load_observations({**cfg, "GENERATE_DATA": False})
                except (OSError, KeyError, ValueError) as error:
                    problems.append(f"{model}: {error}")
            if obs is not None:
                if kind == "lc":
                    light_curves(obs, predictions, ax=ax)
                if kind == "sed":
                    sed(obs, predictions, time=time, ax=ax)
            if problems:
                ax.set_facecolor("#fff9e6")
                ax.text(
                    0.02,
                    0.98,
                    "\n".join(problems),
                    ha="left",
                    va="top",
                    fontsize=6,
                    transform=ax.transAxes,
                )
    handles, labels = [], []
    for ax in axes.flat:
        h, l = ax.get_legend_handles_labels()
        for handle, label in zip(h, l):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(4, len(handles)))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig, axes
