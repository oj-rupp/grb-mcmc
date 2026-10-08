"""One synthetic dataset per grid point, shared by every fitted model."""

from copy import deepcopy
from itertools import product
import json
from pathlib import Path
from .config import DEFAULTS, validate
from .data import generate_observations


def generate_sweep(config_path, output):
    master = json.loads(Path(config_path).read_text())
    grid = master["SWEEP_PARAMS"]
    if not grid or any(not isinstance(v, list) or not v for v in grid.values()):
        raise ValueError("SWEEP_PARAMS must contain nonempty lists")
    models = master["FIT_MODELS"]
    if not models or len(models) != len(set(models)):
        raise ValueError("FIT_MODELS must be nonempty and unique")
    base = Path(output).resolve()
    if base.exists():
        raise FileExistsError(f"Sweep output already exists: {base}")
    jobs = []
    for index, values in enumerate(product(*grid.values())):
        truth = {**master.get("TRUE_PARAMS", {}), **dict(zip(grid, values))}
        cfg = {**DEFAULTS, **deepcopy(master)}
        cfg.update(
            TRUE_PARAMS=truth,
            SEED=master.get("SEED", 42) + index,
            DATA_FILENAME="../observations.npz",
            SAMPLES_FILENAME="samples.h5",
            GENERATE_DATA=False,
        )
        for key in ("SWEEP_PARAMS", "FIT_MODELS"):
            cfg.pop(key, None)
        for model_index, model in enumerate(models):
            job = {**cfg, "FIT_MODEL": model}
            validate(job)
            path = (
                base / f"point_{index:04d}" / f"fit_{model_index:02d}" / "config.json"
            )
            jobs.append((path, job))
    # All ordinary config checks happen before creating the output tree.
    base.mkdir(parents=True)
    (base / "master_config.json").write_text(json.dumps(master, indent=2) + "\n")
    for path, cfg in jobs:
        path.parent.mkdir(parents=True)
        data_path = path.parent.parent / "observations.npz"
        if not data_path.exists():
            generate_observations({**cfg, "DATA_FILENAME": str(data_path)})
        path.write_text(json.dumps(cfg, indent=2) + "\n")
    manifest = [str(path.relative_to(base)) for path, _ in jobs]
    (base / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
