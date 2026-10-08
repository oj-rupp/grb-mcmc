# Analysis and plotting

The package separates reusable calculations from plotting and from notebook
choices. Importing modules does not load chains, change global plotting styles,
create figures or run a fit. Plot functions return figures/axes and never call
`show()` automatically. Use `plt.rc_context(...)` in notebooks for local styling.

## Same data, multiple fitted models

```bash
grb-mcmc compare run/fit_00/config.json run/fit_01/config.json --kind corner --output corner.png
grb-mcmc compare run/fit_00/config.json run/fit_01/config.json --kind lc --output lc.png
grb-mcmc compare run/fit_00/config.json run/fit_01/config.json --kind sed --time 1 --output sed.png
```

Corner plots align columns by parameter name; use `--parameters p log_E52`
to select a common subset. `--orientation` supports `bottom_left`, `bottom_right`,
`top_left`, `top_right`. The same implementation serves single and multiple
models. It uses KDE mass contours at 68%/95%; these are enclosed 2D masses, not
one-dimensional sigma intervals. KDE work uses up to 2,000 deterministic
subsampled points, with padded integration bounds. Singular KDEs fall back to
scatter. Numerical summaries are not subsampled.

LC/SED curves use the highest-posterior **retained** sample, not a 
median. SED data use the nearest observation within 25% of the requested time and show the
actual observation time. They are not asserted to be simultaneous. Negative flux
observations or error bars crossing zero forces the flux axis scale to be linear.

The higher-level comparison requires equal injected parameters, and, for LC/SEDs,
identical observations. Lower-level functions accept explicit arrays if another
comparison is intended. Repeated fits of the same model can be given distinct
labels through the Python corner API.

```python
from grb_mcmc.analysis import posterior
from grb_mcmc.plotting import aligned_posteriors, corner_plot

items = {'run A': posterior('first.h5'), 'run B': posterior('second.h5')}
data, names = aligned_posteriors(items, names=['p', 'log_E52'])
fig, axes = corner_plot(data, names, truth={'p': 2.5, 'log_E52': 0.},
                        orientation='top_right')
fig.savefig('comparison.png')
```

## Sweep grids

```bash
grb-mcmc grid runs/sweep --x log_n_0 --y log_E52 --kind hist --parameters p --output hist.png
grb-mcmc grid runs/sweep --x log_n_0 --y log_E52 --kind contour --parameters p log_E52 --output contours.png
grb-mcmc grid runs/sweep --x log_n_0 --y log_E52 --kind lc --output light-curves.png
grb-mcmc grid runs/sweep --x log_n_0 --y log_E52 --kind sed --time 1 --output spectra.png
```

Use `--models ...` to select fit models. `--truth-models` evaluates injected
parameters for LC/SED grids without requiring chains. All layouts read the
saved configs, not rounded directory names or global variables. Missing/short
chains are marked. If an unplotted third sweep dimension creates duplicate
cells, the function raises instead of silently selecting an arbitrary run.
These grids target the new sweep directory format. Use `compare` or Python
functions for older directory layouts.

## Older chains

```bash
grb-mcmc compare path/to/original_config.json --discard 1000 --output old-corner.png
```

The original config provides `FREE_PARAMS` insertion order. The explicit discard
count is required because the old chain has no authoritative burn-in metadata.
Both maximum-posterior selection and plotting discard the same samples. Nothing
is log-transformed automatically. For relocated old runs, if the configured
file is missing, an exact matching basename beside the config is used.

Old chains remain read-only here and cannot be resumed by the new runner.
The original likelihood/prior conventions and implementation would have to be
verified before enabling continuation; merely attaching metadata is insufficient.

## Degeneracies

```python
from grb_mcmc.degeneracy import pair_regression, joint_alignment, feature_slope

pair = pair_regression(samples, names, 'log_eps_B', 'log_n_0')
expected = feature_slope('log_eps_B', 'log_n_0', 'log_nu_a')
report = joint_alignment(samples, names, standardize=False)
```

The feature matrix is defined once in `degeneracy.py`. Its GS02 scaling
assumptions, including fixed p, limit physical interpretation for other models.
The report contains the primary PCA direction, explained-variance fraction,
isolated feature alignments and the selected joint-subspace alignment.
`standardize=True` transforms both samples **and theoretical vectors** into
the standardized coordinates. This corrects the mixed-coordinate comparison
in the supplied script. Raw log coordinates are the default; choice of metric
changes the question and must be reported. Alignment is descriptive, not a
model-selection statistic. Regression reports slope/intercept/R² without
IID-based uncertainty or p-values for correlated MCMC samples.

The small GS02 example fits only p and energy. Joint physical degeneracies
require all four log parameters; use full research chains for that calculation.
