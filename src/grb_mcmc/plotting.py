"""Reusable plotting primitives. Functions return figures/axes and never show on import."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from .models import evaluate, get_model

LABELS = {
    "p": r"$p$",
    "log_eps_e": r"$\log_{10}\epsilon_e$",
    "log_eps_B": r"$\log_{10}\epsilon_B$",
    "log_n_0": r"$\log_{10}n_0$",
    "log_E52": r"$\log_{10}E_{52}$",
}
COLORS = {
    "gs02_smooth": "#5845d3",
    "VegasAfterglow": "#be0000",
    "PyFRS": "#007a3f",
    "afterglowpy": "#db8c00",
}


def color_for(name, index=0, styles=None):
    style = (styles or {}).get(name, {})
    return style.get(
        "line_color",
        style.get("colors", [None, COLORS.get(name, f"C{index % 10}")])[-1],
    )


def density_contours(ax, x, y, color, fractions=(0.68, 0.95), max_samples=2000):
    """KDE contours with mass thresholds on a padded grid; explicit scatter fallback.

    Evenly spaced deterministic subsampling bounds plotting cost only; fitting
    and numerical summaries always use all retained samples.
    """
    x, y = np.asarray(x), np.asarray(y)
    indices = np.linspace(0, len(x) - 1, min(len(x), max_samples), dtype=int)
    xy = np.vstack([x[indices], y[indices]])
    try:
        kde = gaussian_kde(xy)
    except (np.linalg.LinAlgError, ValueError):
        ax.scatter(xy[0], xy[1], s=2, alpha=0.2, color=color)
        return
    limits = []
    for a in (x, y):
        padding = max(np.std(a) * 2, np.ptp(a) * 0.05, 1e-6)
        limits.append((a.min() - padding, a.max() + padding))
    xx, yy = np.meshgrid(*(np.linspace(*limit, 80) for limit in limits))
    z = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    descending = np.sort(z.ravel())[::-1]
    mass = np.cumsum(descending) / descending.sum()
    levels = np.unique(
        [descending[min(np.searchsorted(mass, f), len(mass) - 1)] for f in fractions]
    )
    levels = levels[levels < z.max()]
    if levels.size:
        ax.contourf(
            xx, yy, z, levels=np.r_[levels, z.max()], colors=[color], alpha=0.18
        )
        ax.contour(xx, yy, z, levels=levels, colors=[color], linewidths=1)


def corner_plot(
    model_data,
    names,
    truth=None,
    styles=None,
    orientation="bottom_left",
    show_95=True,
    figsize=None,
    priors=None,
):
    """One implementation for single/multiple models and all four orientations.

    All arrays must already be ordered to ``names``; use aligned_posteriors for
    chains whose parameter orders differ. Coordinates are never auto-transformed.
    """
    if orientation not in ("bottom_left", "bottom_right", "top_left", "top_right"):
        raise ValueError("Unknown corner orientation")
    if not model_data or not names:
        raise ValueError("Need model samples and parameter names")
    for data in model_data.values():
        if (
            np.ndim(data) != 2
            or data.shape[1] != len(names)
            or len(data) < 2
            or not np.isfinite(data).all()
        ):
            raise ValueError("Expected finite sample arrays with one column per name")
    n = len(names)
    truth = truth or {}
    fig, axes = plt.subplots(n, n, squeeze=False, figsize=figsize or (2.4 * n, 2.4 * n))
    limits = []
    for i, name in enumerate(names):
        a = np.concatenate([data[:, i] for data in model_data.values()])
        low, high = float(a.min()), float(a.max())
        if name in truth:
            low = min(low, truth[name])
            high = max(high, truth[name])
        pad = max((high - low) * 0.06, 1e-6)
        limits.append((low - pad, high + pad))
    handles = []
    for row in range(n):
        for col in range(n):
            x = n - 1 - col if orientation.endswith("right") else col
            y = n - 1 - row if orientation.startswith("top") else row
            ax = axes[row, col]
            if x > y:
                ax.set_visible(False)
                continue
            for index, (name, samples) in enumerate(model_data.items()):
                color = color_for(name, index, styles)
                if x == y:
                    ax.hist(
                        samples[:, x],
                        bins=30,
                        density=True,
                        histtype="step",
                        color=color,
                        label=name,
                    )
                    ax.axvline(np.median(samples[:, x]), color=color, ls=":", lw=1)
                else:
                    density_contours(
                        ax,
                        samples[:, x],
                        samples[:, y],
                        color,
                        (0.68, 0.95) if show_95 else (0.68,),
                    )
                    ax.plot(
                        np.median(samples[:, x]),
                        np.median(samples[:, y]),
                        "o",
                        color=color,
                        ms=3,
                    )
                if row == 0 and col == 0:
                    handles.append(plt.Line2D([], [], color=color, label=name))
            if x == y:
                ax.set_yticks([])
                if names[x] in truth:
                    ax.axvline(truth[names[x]], color="black", ls="--", lw=1)
                if priors and names[x] in priors:
                    for bound in priors[names[x]]:
                        ax.axvline(bound, color="gray", ls=":", lw=0.7)
            else:
                if names[x] in truth and names[y] in truth:
                    ax.plot(truth[names[x]], truth[names[y]], "*", color="black", ms=7)
                ax.set_ylim(limits[y])
                ax.set_ylabel(LABELS.get(names[y], names[y]))
            ax.set_xlim(limits[x])
            ax.set_xlabel(LABELS.get(names[x], names[x]))
    # A figure legend works for every orientation, including an inactive top-left.
    handles = [
        plt.Line2D([], [], color=color_for(name, i, styles), label=name)
        for i, name in enumerate(model_data)
    ]
    fig.legend(handles=handles, loc="upper center", ncol=min(4, len(handles)))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig, axes


def aligned_posteriors(items, names=None):
    """Map independently loaded posteriors to explicitly named common columns."""
    if not items:
        raise ValueError("No posteriors")
    if names is None:
        names = list(next(iter(items.values()))["names"])
    return {
        label: item["samples"][:, [item["names"].index(n) for n in names]]
        for label, item in items.items()
    }, names


def light_curves(obs, predictions=None, ax=None, styles=None):
    """Overlay observed bands and {model: parameter_dict} point-estimate curves."""
    if ax is None:
        _, ax = plt.subplots()
    signed = False
    for index, (label, band) in enumerate(obs.items()):
        color = f"C{index % 10}"
        ax.errorbar(
            band["t_arr"],
            band["Fnu"],
            yerr=band["sigma"],
            fmt="o",
            ms=3,
            color=color,
            label=label,
        )
        signed |= bool(np.any(band["Fnu"] - band["sigma"] <= 0))
        t = np.geomspace(np.min(band["t_arr"]), np.max(band["t_arr"]), 100)
        for j, (model, params) in enumerate((predictions or {}).items()):
            f = evaluate(get_model(model), t, band["nu"], params)
            ax.plot(
                t,
                f,
                color=color,
                ls=["-", "--", ":", "-."][j % 4],
                label=model if index == 0 else None,
            )
    ax.set(
        xscale="log",
        yscale="linear" if signed else "log",
        xlabel="Time (days)",
        ylabel="Flux density (mJy)",
    )
    return ax


def sed(
    obs, predictions, time=1.0, tolerance=0.25, ax=None, styles=None, show_breaks=False
):
    """Nearest observations within relative time tolerance; annotate actual times.

    These data are not interpolated into a simultaneous SED. Model curves are
    evaluated exactly at the requested time.
    """
    if time <= 0 or tolerance < 0:
        raise ValueError("Invalid time or tolerance")
    if ax is None:
        _, ax = plt.subplots()
    signed = False
    for label, band in obs.items():
        k = np.argmin(abs(band["t_arr"] - time))
        actual = band["t_arr"][k]
        if abs(actual - time) / time <= tolerance:
            ax.errorbar(
                band["nu"],
                band["Fnu"][k],
                yerr=band["sigma"][k],
                fmt="o",
                ms=4,
                label=f"{label} ({actual:.3g} d)",
            )
            signed |= bool(band["Fnu"][k] - band["sigma"][k] <= 0)
    frequencies = [b["nu"] for b in obs.values()]
    grid = np.geomspace(min(frequencies) / 2, max(frequencies) * 2, 100)
    for index, (name, params) in enumerate(predictions.items()):
        model = get_model(name)
        # Per-frequency calls respect the generic one-band adapter contract.
        flux = [evaluate(model, np.array([time]), nu, params)[0] for nu in grid]
        color = color_for(name, index, styles)
        ax.plot(grid, flux, color=color, label=name)
        if show_breaks and name == "gs02_smooth":
            _, *breaks = model(np.array([time]), np.array([grid[0]]), **params)
            for label, value in zip(("nu_a", "nu_m", "nu_c"), breaks):
                ax.axvline(float(value[0]), ls=":", color=color, alpha=0.5, label=label)
    ax.set(
        xscale="log",
        yscale="linear" if signed else "log",
        xlabel="Frequency (Hz)",
        ylabel="Flux density (mJy)",
    )
    ax.set_title(f"Model SED at {time:g} d")
    return ax
