# GRB MCMC

A Python framework for fitting multi-band light curves and running synthetic
parameter-recovery experiments with interchangeable gamma-ray burst (GRB) models.

The inference layer is independent of the emission calculation: provide a
Python function that predicts flux density, specify sampled and fixed
parameters, and run the same fitting workflow with different models. The Markov
Chain Monte Carlo (MCMC) process is done indepdently by the `ecmee` Python package; this
framework exists to streamline `emcee` specifically for comparing model fitting between GRB models.

Currently, four GRB models have wrapper support already integrated: the analytic equations
of Granot and Sari (2002) (henceforth referred to as GS02) for ISM and Spectrum 1 assumed, afterglowpy, VegasAfterglow, and 
PyFRS. This codebase also includes shared analysis plots and Physics validation across parameter space
remains the user's responsibility, although there are some diagnostic functions in `physics.py`.

## Features

- Any number of named parameters with uniform bounded priors.
- Importable model adapters, each with `(flux, *extras)` returns.
- Real observations at irregular times or reproducible synthetic datasets.
- Linear-flux Gaussian and approximate log-flux Gaussian likelihoods.
- Persistent `emcee` HDF5 chains, explicit continuation, and overwrite protection.
- One dataset per sweep point shared across fitted models.
- JSON/CSV summaries, corner overlays, light curve (LC)/spectral energy distribution comparisons (SED) and sweep grids.
- Shared band-regime and degeneracy diagnostics, plus an import-only example notebook.
- Serial execution by default; optional process workers using `spawn` for computing cluster use.

## Install and try it

From the repository directory:

```bash
python -m venv .venv
source .venv/bin/activate
# Windows: .venv\Scripts\activate
python -m pip install -e ".[plot,dev]"
grb-mcmc fit examples/toy.json
grb-mcmc summarize runs/toy/samples.h5
grb-mcmc plot runs/toy/samples.h5 --kind trace --output runs/toy/trace.png
grb-mcmc plot runs/toy/samples.h5 --kind corner --output runs/toy/corner.png
python -m pytest
```

`python -m grb_mcmc.cli` is equivalent to `grb-mcmc`.
The example is an executable demonstration; its step count is not a
recommendation for scientific runs. Paths in fit configs are resolved relative
to the JSON file, so a run does not depend on the working directory.

## Fit your own model

Place an adapter in an importable Python module, e.g. `my_models.py` in this
repository when running from its root, or install your model package:

```python
import numpy as np

def flux_density(t_days, nu_hz, **params):
    # Convert units and named parameters here, call your model engine,
    # and return flux density in mJy, one value for each input time.
    # nu_hz is a one-element array; each band is evaluated separately.
    return your_engine(t_days, nu_hz, **params)
```

Set `FIT_MODEL` to `"my_models:flux_density"`. The adapter may return an array
or `(flux_array, extra_outputs, ...)`. Accepted array shapes squeeze to
`(number_of_times,)`; one-point bands are supported. Do not return a plain list
of fluxes plus extras: use a tuple for that convention.

Parameters are passed unchanged. In particular, a name such as `log_E52`
is **not** automatically exponentiated. Adapters must define and implement
those conventions, including conversions to seconds, cgs energy, distance,
and any model-specific angle conventions. See [model adapters](docs/models.md).

## Configuration

Start with [examples/toy.json](examples/toy.json).

| Key | Meaning |
| --- | --- |
| `FIT_MODEL` | A built-in model name or an importable `module:function` |
| `FREE_PARAMS` | Sampled names and **initial guesses**; JSON order is chain column order |
| `FIXED_PARAMS` | Parameters held constant; cannot overlap `FREE_PARAMS` names |
| `PRIORS` | `[lower, upper]` for each sampled parameter; matched by name |
| `LIKELIHOOD` | `linear_flux` or `log_flux` |
| `DATA_FILENAME` | Observations NPZ file |
| `SAMPLES_FILENAME` | Output chain HDF5 file |
| `NWALKERS` | Number walkers chains sampling during MCMC run |
| `BURN_STEPS`, `STEPS` | Initial burn-in and desired total production length |
| `SEED`, `WORKERS`, `PROGRESS` | Reproducibility and execution settings |
| `GENERATE_DATA` | Generate only if data are missing |
| `GENERATOR_MODEL`, `TRUE_PARAMS` | Synthetic model and injected parameters (to which Gaussian noise is later added), separate from guesses |
| `BANDS` | Synthetic grid: `[label, frequency_Hz, start_days, end_days, count]` |
| `FRACTIONAL_ERROR` | Synthetic Gaussian standard deviation divided by model flux |

The default likelihood is `log_flux` to sample evenly across all frequencies.
The examples explicitly choose `linear_flux`, which matches the
synthetic generator's additive Gaussian noise. Changing likelihoods changes
the statistical model; rerun comparisons consistently. No bands are 
reweighted: every observation contributes once, so densely sampled bands
contribute more terms. Priors are uniform in the **sampled coordinates**;
a uniform prior on `log_E52` is a log-uniform prior on `E52`.

## Physical Parameters

| Parameter | Variable Name | Meaning |
| --- | --- | --- |
| $E_{52}$ | `E52` | Total isotropic energy (erg/$10^{52}$) |
| $\epsilon_{\text{B}}$ | `eps_B` | Fraction of energy in magnetic field |
| $\epsilon_{\text{e}}$ | `eps_e` | Fraction of energy in electrons |
| $n_{0}$ | `n_0` | Circumburst particle number density ($\text{cm}^{-3}$) |
| $p$ | `p` | Characterizes slope of electron distribution in different physical regimes [1] |
| $z$ | `z` | Redshift|
| $d_{\text{L}28}$ | `dL28` | Luminosity distance scaled by $10^{28}$ cm |
| $\theta_{\text{c}}$ | `thetaCore` | Jet core opening angle (radians) |
| $\theta_{\text{o}}$ | `thetaObs` | Observing angle of jet (radians) |
| $\Gamma_{0}$ | `Gamma0` | Bulk Lorentz factor (constant during coasting phase) |


## Real observations

Save one set of arrays per band. Different bands may have different lengths
and irregular sampling. Time is in days, frequency in Hz, and flux and its
1-$\sigma$ uncertainty in mJy:

```python
import numpy as np
from grb_mcmc.data import save_observations

save_observations("observations.npz", {
    "radio": {
        "t_arr": np.array([0.4, 1.7, 5.2]),
        "nu": 1.0e10,
        "Fnu": np.array([0.12, 0.08, 0.03]),
        "sigma": np.array([0.01, 0.015, 0.01]),
    },
})
```

Use a fit config with `GENERATE_DATA: false`. `GENERATOR_MODEL`, `TRUE_PARAMS`
and `BANDS` are unnecessary when a self-contained observations file exists, and changing 
these settings in the JSON file will not change the aleady-generated observations.
All bands in the file are fitted. The loader also reads legacy files containing
only `Fnu_<band>` and `sigma_<band>`, using `BANDS` to reconstruct their grids.
It cannot recover irregular times that were never saved.

Negative observed fluxes are allowed with `linear_flux`; the emission model
must still predict positive finite flux. `log_flux` requires positive observed
flux and uses `sigma_log = sigma / (F_obs * ln(10))`, a small-error
approximation. Upper limits, correlated calibration errors, extinction and
bandpass integration are not implemented by the inference core. Do not encode
upper limits as Gaussian detections.

## Continue a run

Raise `STEPS` in the same config, then:

```bash
grb-mcmc fit examples/toy.json --resume
```

`STEPS` is a target total, not an increment. Burn-in is retained in the HDF5
file and discarded exactly once in summaries and plots. Existing files are
never reset by a new fit. Resume checks observations, parameter order, and
configuration, while allowing changed paths, worker count, progress display
and target length. A chain with no saved samples restarts initialization.
Do not run concurrent writers against the same chain.

A resume fingerprint does not detect edits inside an external model package.
Keep model code and dependency versions unchanged when continuing a chain.
For reproducible research, record your repository commit and environment
(e.g. `python -m pip freeze > environment-lock.txt`) alongside results.

## Sweeps and cluster jobs

```bash
grb-mcmc sweep examples/toy_sweep.json --output runs/sweep
grb-mcmc job runs/sweep/manifest.json 0
grb-mcmc collect runs/sweep --output runs/sweep/summary.csv
```

`SWEEP_PARAMS` varies injected truth; `FREE_PARAMS` still specifies fitting
initial guesses. Each point has its own deterministic seed (`SEED + index`),
and all fit models at that point share the same observations. Sweep generation
refuses an existing destination to avoid replacing data or job definitions.
Numeric directory IDs avoid collisions from rounded parameter names.

The manifest uses zero-based job indices. Run indices 0–3 for the supplied
four-job example, or dispatch them with [examples/slurm.sh](examples/slurm.sh).
Set `WORKERS` no larger than the CPUs allocated per job. Start with one worker
for cheap models, where process overhead may exceed any speedup.

Collection keeps missing and failed jobs as rows with a status/error, rather
than silently excluding them. It does not automatically run missing jobs.
The collector's `chi2_at_max_posterior` is the sum of squared standardized
residuals at the best saved posterior sample. It is not Bayesian evidence.

## Diagnostics and scope

Summaries report 16th/50th/84th percentiles, acceptance fraction and, when
estimable, autocorrelation times. `autocorr_length_check_passed` requires
production length at least 50 times every estimated autocorrelation time.
This is a heuristic length check, not proof of convergence. Inspect trace
plots, compare independent runs, and check stability as runs are extended.
See the [emcee autocorrelation tutorial](https://emcee.readthedocs.io/en/stable/tutorials/autocorr/).

See [analysis and plotting](docs/analysis.md), [model conventions](docs/models.md),
[second cleanup migration](docs/migration-v2.md), and [validation](docs/validation.md).
The [first-cleanup migration](docs/migration.md) is retained as historical context.

## Physical-model quickstart

GS02 needs no extra engine. Other wrappers are optional:

```bash
python -m pip install -e ".[plot,dev,models]"
grb-mcmc fit examples/gs02.json
grb-mcmc compare examples/gs02.json --kind corner --output runs/gs02/corner.png
grb-mcmc compare examples/gs02.json --kind lc --output runs/gs02/lc.png
grb-mcmc compare examples/gs02.json --kind sed --time 1 --output runs/gs02/sed.png
grb-mcmc check-bands examples/gs02.json
```

Or run `conda env create -f environment.yml` from the repository root. This
file installs the same package extras, avoiding a second dependency list.
The supplied GS02 demo fits p and energy with other physical values fixed;
it is a short functional demonstration, not a full five-parameter recovery study.
`examples/grb_sweep.json` illustrates four-model cross-fitting; these models
have different approximations and distance/parameter support, documented in
[model conventions](docs/models.md).

## Restart without a separate script

```bash
grb-mcmc extend examples/gs02.json --steps 500
grb-mcmc extend examples/gs02.json --until-stable --chunk 500 --max-chunks 20
```

The first appends exactly the requested number of steps once the run has entered
production. The second is bounded and stops only after length >= 50 tau and
less than 1% change in every estimated tau across consecutive chunks. Its
stopping reason is printed; hitting the cap is not reported as convergence.
Neither edits the config file. `fit --resume` still uses the config's target
length, so use `extend` when expressing an increment.

## What remains to be added

The main improvements for this codebase lie in optimization. Some of the MCMC process is bottlenecked by the `emcee` package itself,
but it remains to be tested how adjusting wrapper calls and other function calls improves preformance. Additionally, I also hope to
incorporate more streamlined integration into different HPC job schedulers, such as Slurm and Sun Grid Engine (SGE), as well as automated file-reading for real GRB datasets.

## References

1. Granot, J., & Sari, R. (2002). The Shape of Spectral Breaks in Gamma-Ray Burst Afterglows. The Astrophysical Journal, 568(2), 820. GS02

## Acknowledgements

This codebase was primarily developed while I worked with Professor Tanmoy Laskar at the University of Utah REU during the summer of 2026. I am incredibly grateful for Professor Laskar's support and mentorship during my development of this codebase.