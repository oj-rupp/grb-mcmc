"""Uniform box priors and independent Gaussian flux likelihoods."""

import numpy as np
from .models import evaluate


def log_prior(theta, names, priors):
    theta = np.asarray(theta)
    if theta.shape != (len(names),) or not np.isfinite(theta).all():
        return -np.inf
    return (
        0.0
        if all(priors[n][0] < x < priors[n][1] for n, x in zip(names, theta))
        else -np.inf
    )


def residuals(theta, model, names, fixed, obs, likelihood):
    params = {**fixed, **dict(zip(names, theta))}
    residual, errors = [], []
    for band in obs.values():
        predicted = evaluate(model, band["t_arr"], band["nu"], params)
        if not np.isfinite(predicted).all() or (predicted <= 0).any():
            return None, None
        if likelihood == "log_flux":
            sigma = band["sigma"] / (band["Fnu"] * np.log(10))
            difference = np.log10(band["Fnu"]) - np.log10(predicted)
        elif likelihood == "linear_flux":
            sigma = band["sigma"]
            difference = band["Fnu"] - predicted
        else:
            raise ValueError(f"Unknown likelihood: {likelihood}")
        residual.append(difference / sigma)
        errors.append(sigma)
    return np.concatenate(residual), np.concatenate(errors)


def log_likelihood(theta, model, names, fixed, obs, likelihood):
    r, sigma = residuals(theta, model, names, fixed, obs, likelihood)
    if r is None:
        return -np.inf
    with np.errstate(over="ignore", invalid="ignore"):
        value = -0.5 * np.sum(r**2 + np.log(2 * np.pi) + 2 * np.log(sigma))
    return float(value) if np.isfinite(value) else -np.inf


def log_posterior(theta, model, names, fixed, obs, likelihood, priors):
    lp = log_prior(theta, names, priors)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, model, names, fixed, obs, likelihood)
