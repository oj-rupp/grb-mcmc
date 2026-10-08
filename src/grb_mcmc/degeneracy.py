"""
Degeneracy diagnostics in explicit sampled coordinates. This section remains to be fully tested.
"""

import numpy as np
from scipy.stats import linregress

PARAMETERS = ("log_eps_e", "log_eps_B", "log_n_0", "log_E52")
FEATURES = ("log_nu_a", "log_nu_m", "log_nu_c", "log_Fnu_max")
# Matrix is from GS02 dervied degeneracies; it assumes GS02 scaling regime/fixed p.
FEATURE_MATRIX = np.array(
    [
        [5 / 6, 11 / 12, 1 / 4, -1 / 2],
        [-5 / 2, -5 / 4, -5 / 4, 1 / 2],
        [25 / 6, 25 / 12, 3 / 4, -3 / 2],
        [-5 / 6, -5 / 12, 1 / 4, 3 / 2],
    ]
)


def feature_slope(x, y, feature):
    vector = FEATURE_MATRIX[:, FEATURES.index(feature)]
    dx = vector[PARAMETERS.index(x)]
    return float(vector[PARAMETERS.index(y)] / dx) if dx else np.inf


def pair_regression(samples, names, x, y):
    """OLS as a geometric description; no IID significance claims for MCMC draws."""
    result = linregress(samples[:, names.index(x)], samples[:, names.index(y)])
    return dict(
        slope=float(result.slope),
        intercept=float(result.intercept),
        r_squared=float(result.rvalue**2),
    )


def joint_alignment(
    samples, names, features=("log_nu_a", "log_nu_c"), standardize=False
):
    """PCA and feature vectors use the same metric, optionally standardized.

    Reports alignment of the first PC with the selected feature subspace.
    Results depend on metric and on the applicability of the supplied GS02 matrix.
    """
    x = np.asarray(samples)[:, [names.index(p) for p in PARAMETERS]]
    if len(x) < 5 or not np.isfinite(x).all() or np.any(np.std(x, axis=0) == 0):
        raise ValueError(
            "Need finite, nonconstant samples of all four physical log parameters"
        )
    if not features or len(set(features)) != len(features):
        raise ValueError("Select nonempty, unique feature names")
    scale = np.std(x, axis=0) if standardize else np.ones(4)
    _, singular, vt = np.linalg.svd((x - x.mean(axis=0)) / scale, full_matrices=False)
    direction = vt[0]
    # Transform theory exactly as the samples before normalization/projection.
    vectors = FEATURE_MATRIX / scale[:, None]
    vectors = vectors / np.linalg.norm(vectors, axis=0)
    basis, _ = np.linalg.qr(vectors[:, [FEATURES.index(f) for f in features]])
    return dict(
        metric="standardized" if standardize else "raw_log",
        features=list(features),
        primary_direction=direction.tolist(),
        first_pc_variance_fraction=float(singular[0] ** 2 / np.sum(singular**2)),
        joint_alignment=float(np.clip(np.linalg.norm(basis.T @ direction), 0, 1)),
        isolated_alignments={
            f: float(abs(direction @ vectors[:, i])) for i, f in enumerate(FEATURES)
        },
    )
