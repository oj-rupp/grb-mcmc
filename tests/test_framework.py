import json
from pathlib import Path
import subprocess
import sys
import emcee
import numpy as np
import pytest
from grb_mcmc.config import load_config
from grb_mcmc.data import load_observations, save_observations, validate_observations
from grb_mcmc.inference import log_prior, log_likelihood
from grb_mcmc.models import evaluate
from grb_mcmc.runner import run
from grb_mcmc.sweep import generate_sweep
from grb_mcmc.analysis import summarize, collect


@pytest.fixture
def config(tmp_path):
    template = Path(__file__).parents[1] / "examples/toy.json"
    cfg = json.loads(template.read_text())
    cfg.update(
        DATA_FILENAME="observations.npz",
        SAMPLES_FILENAME="samples.h5",
        BURN_STEPS=10,
        STEPS=30,
    )
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg))
    return path


def edit(path, **values):
    cfg = json.loads(path.read_text())
    cfg.update(values)
    path.write_text(json.dumps(cfg))


def test_named_prior_order_and_dimensions():
    priors = {"b": [10, 20], "a": [-1, 1]}
    assert log_prior([0, 15], ["a", "b"], priors) == 0
    assert log_prior([15, 0], ["a", "b"], priors) == -np.inf
    assert log_prior([0], ["a", "b"], priors) == -np.inf
    assert log_prior([np.nan, 15], ["a", "b"], priors) == -np.inf
    assert log_prior([-1, 15], ["a", "b"], priors) == -np.inf
    assert log_prior([0.5], ["x"], {"x": [0, 1]}) == 0


def test_likelihood_matches_hand_calculation():
    obs = {
        "one": dict(
            t_arr=np.array([1.0]), nu=1e14, Fnu=np.array([3.0]), sigma=np.array([0.5])
        )
    }

    def model(t, nu, **params):
        return np.array([[2.0]])

    expected = -0.5 * (4 + np.log(2 * np.pi * 0.5**2))
    assert log_likelihood([], model, [], {}, obs, "linear_flux") == pytest.approx(
        expected
    )
    err = 0.5 / (3 * np.log(10))
    expected_log = -0.5 * ((np.log10(3 / 2) / err) ** 2 + np.log(2 * np.pi * err**2))
    assert log_likelihood([], model, [], {}, obs, "log_flux") == pytest.approx(
        expected_log
    )
    assert evaluate(
        lambda *a, **k: (np.array([[2.0]]), None), np.array([1.0]), 1e14, {}
    ).shape == (1,)


def test_data_round_trip_irregular_and_signed(tmp_path):
    obs = {
        "radio": dict(
            t_arr=np.array([0.3, 1.7, 4.1]),
            nu=1e10,
            Fnu=np.array([1.0, -0.1, 0.0]),
            sigma=np.ones(3),
        )
    }
    path = tmp_path / "real.npz"
    save_observations(path, obs)
    recovered = load_observations(dict(DATA_FILENAME=path, LIKELIHOOD="linear_flux"))
    np.testing.assert_array_equal(recovered["radio"]["t_arr"], obs["radio"]["t_arr"])
    with pytest.raises(ValueError):
        validate_observations(recovered, "log_flux")
    with pytest.raises(FileExistsError):
        save_observations(path, obs)


def test_legacy_npz(config):
    cfg = load_config(config)
    flat = {}
    for label, nu, start, stop, count in cfg["BANDS"]:
        flat["Fnu_" + label] = np.ones(count)
        flat["sigma_" + label] = np.ones(count) * 0.1
    np.savez(cfg["DATA_FILENAME"], **flat)
    obs = load_observations(cfg)
    assert len(obs["optical"]["t_arr"]) == 25


def test_resume_reproduces_uninterrupted_chain_and_protects(config):
    path = run(config)
    before = emcee.backends.HDFBackend(str(path), read_only=True).get_chain().copy()
    with pytest.raises(FileExistsError):
        run(config)
    edit(config, STEPS=60)
    run(config, resume=True)
    resumed = emcee.backends.HDFBackend(str(path), read_only=True).get_chain()
    np.testing.assert_array_equal(resumed[: len(before)], before)
    edit(config, SAMPLES_FILENAME="full.h5")
    full = run(config)
    np.testing.assert_array_equal(
        resumed, emcee.backends.HDFBackend(str(full), read_only=True).get_chain()
    )
    edit(config, LIKELIHOOD="log_flux")
    with pytest.raises(ValueError, match="Resume refused"):
        run(config, resume=True)


def test_short_chain_is_not_declared_converged(config):
    path = run(config)
    summary = summarize(path)
    assert summary["autocorr_length_check_passed"] is False
    assert summary["production_steps"] == 30
    assert set(summary["parameters"]) == {"alpha", "log_amplitude"}


def test_recovery(config):
    edit(config, BURN_STEPS=100, STEPS=500)
    summary = summarize(run(config))
    assert abs(summary["parameters"]["alpha"]["median"] - 1.2) < 0.1
    assert abs(summary["parameters"]["log_amplitude"]["median"]) < 0.1


def test_sweep_shared_data_and_collection(config, tmp_path):
    cfg = json.loads(config.read_text())
    cfg.update(
        SWEEP_PARAMS={"alpha": [0.8, 1.2]},
        FIT_MODELS=["toy", "grb_mcmc.models:toy_flux"],
    )
    config.write_text(json.dumps(cfg))
    output = tmp_path / "sweep"
    jobs = generate_sweep(config, output)
    assert len(jobs) == 4
    assert len(list(output.glob("point_*/observations.npz"))) == 2
    rows = collect(output)
    assert all(row["status"] == "unavailable" for row in rows)
    for job in jobs[:2]:
        run(output / job)
    rows = collect(output)
    assert sum(row["status"] == "complete" for row in rows) == 2
    assert rows[0]["chi2_at_max_posterior"] >= 0
    assert rows[0]["true_alpha"] == 0.8
    with pytest.raises(FileExistsError):
        generate_sweep(config, output)


def test_spawn_workers_and_cli(config):
    edit(config, WORKERS=2, STEPS=4, BURN_STEPS=2)
    subprocess.run(
        [sys.executable, "-m", "grb_mcmc.cli", "fit", str(config)],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    output = subprocess.run(
        [
            sys.executable,
            "-m",
            "grb_mcmc.cli",
            "summarize",
            str(config.parent / "samples.h5"),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert json.loads(output.stdout)["production_steps"] == 4


def test_resume_detects_data_and_parameter_order_changes(config):
    run(config)
    cfg = json.loads(config.read_text())
    cfg["FREE_PARAMS"] = dict(reversed(list(cfg["FREE_PARAMS"].items())))
    config.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="Resume refused"):
        run(config, resume=True)
    cfg["FREE_PARAMS"] = dict(reversed(list(cfg["FREE_PARAMS"].items())))
    config.write_text(json.dumps(cfg))
    data_path = config.parent / "observations.npz"
    with np.load(data_path) as source:
        data = {k: source[k].copy() for k in source.files}
    data["Fnu_optical"][0] *= 1.01
    np.savez(data_path, **data)
    with pytest.raises(ValueError, match="Resume refused"):
        run(config, resume=True)
