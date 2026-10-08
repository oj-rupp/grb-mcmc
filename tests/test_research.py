"""Regression and integration coverage for the supplied research modules."""

import json
from pathlib import Path
import sys
import subprocess
import numpy as np
import pytest
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import emcee
import h5py
from grb_mcmc.models import get_model, evaluate
from grb_mcmc.analysis import read_chain, posterior
from grb_mcmc.runner import run, extend
from grb_mcmc.physics import band_diagnostics
from grb_mcmc.degeneracy import joint_alignment, FEATURE_MATRIX, PARAMETERS
from grb_mcmc.plotting import aligned_posteriors, corner_plot, light_curves, sed
from grb_mcmc.sweep import generate_sweep
from grb_mcmc.grids import sweep_grid


@pytest.mark.parametrize(
    "name", ["gs02_smooth", "afterglowpy", "VegasAfterglow", "PyFRS"]
)
def test_original_wrapper_regression(name):
    if name != "gs02_smooth":
        pytest.importorskip(name)
    if name == "PyFRS":
        pytest.importorskip("astropy")
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/original_fluxes.json").read_text()
    )
    for nu in fixture["frequencies"]:
        actual = evaluate(
            get_model(name), np.array(fixture["times"]), nu, fixture["params"]
        )
        np.testing.assert_allclose(
            actual, fixture["expected"][name][str(nu)], rtol=1e-7
        )
        assert (actual > 0).all()
    one = evaluate(get_model(name), np.array([1.0]), 1e14, fixture["params"])
    assert one.shape == (1,) and np.isfinite(one).all()


def make_config(tmp_path):
    cfg = json.loads((Path(__file__).parents[1] / "examples/toy.json").read_text())
    cfg.update(
        DATA_FILENAME="obs.npz", SAMPLES_FILENAME="chain.h5", STEPS=30, BURN_STEPS=10
    )
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg))
    return path


def test_extension_and_legacy_consistent_discard(tmp_path):
    path = make_config(tmp_path)
    chain = run(path)
    original = emcee.backends.HDFBackend(str(chain), read_only=True).get_chain()
    result = extend(path, steps=5)
    assert result["total_steps"] == 45
    np.testing.assert_array_equal(
        emcee.backends.HDFBackend(str(chain), read_only=True).get_chain()[:40], original
    )
    result = extend(path, chunk=2, max_chunks=1)
    assert result["stopped_by"] == "max_chunks" and result["total_steps"] == 47
    with h5py.File(chain, "a") as stream:
        del stream.attrs["parameter_names"]
    with pytest.raises(ValueError, match="Legacy chain"):
        read_chain(chain)
    names, cfg, samples, prob, _ = read_chain(chain, path, discard=20)
    assert samples.shape[0] == 27 and len(prob) == 27 * cfg["NWALKERS"]
    item = posterior(chain, path, discard=20)
    expected = samples.reshape(-1, 2)[np.argmax(prob)]
    np.testing.assert_array_equal([item["best_params"][n] for n in names], expected)


def test_standardized_degeneracy_transforms_theory():
    rng = np.random.default_rng(5)
    vector = FEATURE_MATRIX[:, 0]
    samples = rng.normal(size=(2000, 1)) * vector + rng.normal(size=(2000, 4)) * 0.005
    for standardized in [False, True]:
        result = joint_alignment(samples, list(PARAMETERS), standardize=standardized)
        assert result["joint_alignment"] > 0.999
        assert result["isolated_alignments"]["log_nu_a"] > 0.999
    with pytest.raises(ValueError):
        joint_alignment(samples[:, :2], list(PARAMETERS[:2]))


def test_parameter_alignment_and_corner_orientations():
    rng = np.random.default_rng(2)
    sample = rng.normal(size=(150, 2))
    data, names = aligned_posteriors(
        {
            "a": {"samples": sample, "names": ["a", "b"]},
            "b": {"samples": sample[:, ::-1], "names": ["b", "a"]},
        }
    )
    np.testing.assert_array_equal(data["a"], data["b"])
    for orientation in ["bottom_left", "bottom_right", "top_left", "top_right"]:
        fig, axes = corner_plot(
            data, names, truth={"a": 0, "b": 0}, orientation=orientation
        )
        assert sum(ax.get_visible() for ax in axes.flat) == 3
        plt.close(fig)


def test_signed_lc_and_non_simultaneous_sed():
    obs = {
        "one": dict(
            t_arr=np.array([0.2, 1.2]),
            nu=1e14,
            Fnu=np.array([-0.1, 2.0]),
            sigma=np.array([0.2, 0.1]),
        )
    }
    ax = light_curves(obs, {"toy": {}})
    assert ax.get_yscale() == "linear"
    plt.close(ax.figure)
    ax = sed(obs, {"toy": {}}, time=1.0, tolerance=0.25)
    assert any("1.2 d" in text for text in ax.get_legend_handles_labels()[1])
    plt.close(ax.figure)


def test_sweep_grid_all_modes_and_no_import_side_effects(tmp_path):
    path = make_config(tmp_path)
    cfg = json.loads(path.read_text())
    cfg.update(
        SWEEP_PARAMS={"alpha": [1.0, 1.2], "log_amplitude": [0.0]}, FIT_MODELS=["toy"]
    )
    path.write_text(json.dumps(cfg))
    base = tmp_path / "sweep"
    jobs = generate_sweep(path, base)
    for job in jobs:
        run(base / job)
    for kind, params in [
        ("hist", ["alpha"]),
        ("contour", ["alpha", "log_amplitude"]),
        ("lc", []),
        ("sed", []),
    ]:
        fig, axes = sweep_grid(base, "alpha", "log_amplitude", kind, params)
        assert axes.shape == (1, 2)
        plt.close(fig)
    code = "import matplotlib.pyplot as p; before=dict(p.rcParams); import grb_mcmc.plotting, grb_mcmc.grids; assert before==dict(p.rcParams); assert not p.get_fignums()"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_gs02_regime_coverage():
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/original_fluxes.json").read_text()
    )
    params = fixture["params"]
    report = band_diagnostics(
        {"opt": {"nu": 1e14, "t_arr": np.array([0.5, 1.0, 2.0])}}, params
    )
    assert report["bands"]["opt"]["spectrum_1_ordering_at_all_times"]
    assert report["jet_time_days"] > 0


def test_rejects_sampling_ignored_model_parameters(tmp_path):
    from grb_mcmc.config import load_config

    path = make_config(tmp_path)
    cfg = json.loads(path.read_text())
    cfg.update(
        FIT_MODEL="PyFRS", FREE_PARAMS={"dL28": 2.0}, PRIORS={"dL28": [1.0, 3.0]}
    )
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="does not use"):
        load_config(path)
