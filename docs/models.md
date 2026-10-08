# Model adapters

The package includes the four supplied wrappers, loaded only when selected:

| Config name | Implementation | Optional install |
| --- | --- | --- |
| `gs02_smooth` | Supplied Granot and Sari 2002 Spectrum 1 equations | Core package |
| `afterglowpy` | Top-hat jet, `SimpleSpec` | `pip install -e '.[afterglow]'` |
| `VegasAfterglow` | Top-hat jet, ISM, forward radiation | `pip install -e '.[vegas]'` |
| `PyFRS` | Forward shock, k=0, SSC/XIC disabled | `pip install -e '.[pyfrs]'` |

All adapters receive time in days, frequency in Hz, angles in radians, and
return flux density in mJy. The original GS02 equations and engine parameter
mappings are retained. The PyFRS wrapper converts radians to degrees; numerical
engines receive seconds. `log_E52` represents log10 of energy in units of
10^52 erg; `log_n_0` is log10 number density in cm^-3. `log_eps_e` and
`log_eps_B` are base-10 logs of the energy fractions. `dL28` is luminosity
distance in units of 10^28 cm.

## Physical parameters actually used

See [README](/README.md) for physical parameter descriptions.
Every adapter uses `p`, `log_eps_e`, `log_eps_B`, `log_n_0`, `log_E52`, and `z`.

| Adapter | Additional parameters used | Common fixed parameters ignored |
| --- | --- | --- |
| GS02 | `dL28` | `thetaCore`, `thetaObs`, `Gamma0`, `xi_N` |
| afterglowpy | `dL28`, `thetaCore`, `thetaObs`, `xi_N` | `Gamma0` |
| VegasAfterglow | `dL28`, `thetaCore`, `thetaObs`, `Gamma0` | `xi_N` |
| PyFRS | `thetaCore`, `thetaObs`, `Gamma0` | `dL28`, `xi_N` |

A common fixed dictionary stores all parameters and their values. Model
references reject sampled parameters that the wrapper does not use. For new models with 
additional parameters, the author is responsible for declaring an appropriate parameter
set. No adapter substitutes missing required physical values with defaults, so failing
to add all required parameters will result in an error.

**PyFRS distance convention:** the backend computes its own distance
from redshift using the astropy package (`astro.LUD(z)`). It does not use
`dL28`. Cross-model comparisons therefore need compatible distance conventions.

**GS02 regime:** this implementation assumes Spectrum 1 of GS02, `nu_a < nu_m < nu_c`,
and an ISM. `grb-mcmc check-bands CONFIG` reports break ordering at the specified
parameters and times. It is not a proof that the entire posterior lies inside
the regime, and no new regime cut is silently added to the likelihood. The
original transition-time, jet-time and deceleration-time formulae remain
available in `physics.py` as research diagnostics, but they have not been automated
into the framework, as I assume the primary use for this package is to sample more 
developed models than the analytic equations of GS02.

## Custom adapter contract

Provide an importable top-level function:

```python
def flux_density(t_days, nu_hz, **params):
    # nu_hz is a one-element array when called by the inference core.
    # Return one flux in mJy per time, shape (n,), (n,1), or (1,n).
    return your_model(t_days, nu_hz, **params)
```

Set `FIT_MODEL` to `your_package.module:flux_density`. An array or a tuple
`(flux_array, extra_outputs, ...)` is accepted. The core passes parameters
unchanged; custom adapters own all transformations. Only GS02 returns genuine
break-frequency arrays. Other adapters now return flux arrays instead of
fabricated break outputs. Plot code requests breaks only from GS02.

Nonpositive/nonfinite predictions reject a proposal. Shape errors and unexpected
backend exceptions propagate rather than being hidden as normal rejections.
Known domain failures can be converted to correctly shaped NaN arrays inside
the adapter.

GRB model wrappers live under `grb_mcmc.adapters` with distinct module names, preventing
local `afterglowpy.py`, `PyFRS.py` or `VegasAfterglow.py` from shadowing the
installed engines. Optional versions tested here: afterglowpy 0.8.1,
VegasAfterglow 2.0.6, PyFRS 0.1.0.2, with Astropy 8.0.1. The PyFRS extra explicitly
includes Astropy because its package metadata does not declare that import.

References:
1. Granot, J., & Sari, R. (2002). The Shape of Spectral Breaks in Gamma-Ray Burst Afterglows. The Astrophysical Journal, 568(2), 820. GS02
2. [afterglowpy documentation](https://afterglowpy.readthedocs.io/)
3. [VegasAfterglow documentation](https://vegasafterglow.readthedocs.io/en/stable/)
4. [PyFRS source and model conventions](https://github.com/leiwh/PyFRS)

Add the appropriate scientific and software citations for models used in a
publication; numerical regression against the supplied wrappers is not an
independent validation of their physics.
