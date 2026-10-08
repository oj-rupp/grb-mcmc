"""Single command-line entry point for local and cluster workflows."""

import argparse
import csv
import json
from pathlib import Path
from .runner import run, extend
from .analysis import summarize, collect, plot_chain, plot_corner
from .sweep import generate_sweep


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("fit", help="Run or continue one JSON configuration")
    fit.add_argument("config")
    fit.add_argument("--resume", action="store_true")
    sweep = sub.add_parser(
        "sweep", help="Generate synthetic data and independent job configs"
    )
    sweep.add_argument("config")
    sweep.add_argument("--output", required=True)
    job = sub.add_parser(
        "job", help="Run zero-based manifest index (e.g. SLURM array index)"
    )
    job.add_argument("manifest")
    job.add_argument("index", type=int)
    job.add_argument("--resume", action="store_true")
    summary = sub.add_parser("summarize")
    summary.add_argument("chain")
    summary.add_argument("--output")
    collector = sub.add_parser("collect")
    collector.add_argument("directory")
    collector.add_argument("--output", required=True)
    plot = sub.add_parser("plot")
    plot.add_argument("chain")
    plot.add_argument("--kind", choices=["trace", "corner"], default="trace")
    plot.add_argument("--output", required=True)
    extension = sub.add_parser(
        "extend", help="Append steps or use bounded autocorrelation stopping"
    )
    extension.add_argument("config")
    group = extension.add_mutually_exclusive_group(required=True)
    group.add_argument("--steps", type=int)
    group.add_argument("--until-stable", action="store_true")
    extension.add_argument("--chunk", type=int, default=500)
    extension.add_argument("--max-chunks", type=int, default=20)
    grid = sub.add_parser(
        "grid", help="Sweep histograms, contours, light curves or SEDs"
    )
    grid.add_argument("directory")
    grid.add_argument("--x", required=True)
    grid.add_argument("--y", required=True)
    grid.add_argument(
        "--kind", choices=["hist", "contour", "lc", "sed"], default="hist"
    )
    grid.add_argument("--parameters", nargs="*", default=[])
    grid.add_argument("--models", nargs="+")
    grid.add_argument("--time", type=float, default=1.0)
    grid.add_argument("--truth-models", action="store_true")
    grid.add_argument("--output", required=True)
    check = sub.add_parser(
        "check-bands", help="GS02 regime/coverage diagnostics at supplied parameters"
    )
    check.add_argument("config")
    check.add_argument("--output")
    compare = sub.add_parser(
        "compare", help="Named posterior overlays or best-sample LC/SEDs"
    )
    compare.add_argument("configs", nargs="+")
    compare.add_argument("--kind", choices=["corner", "lc", "sed"], default="corner")
    compare.add_argument(
        "--orientation",
        choices=["bottom_left", "bottom_right", "top_left", "top_right"],
        default="bottom_left",
    )
    compare.add_argument("--parameters", nargs="+")
    compare.add_argument("--time", type=float, default=1.0)
    compare.add_argument(
        "--discard", type=int, help="Required explicit burn-in for legacy chains"
    )
    compare.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "fit":
        print(run(args.config, args.resume))
    elif args.command == "sweep":
        print(f"Generated {len(generate_sweep(args.config, args.output))} jobs")
    elif args.command == "job":
        manifest = Path(args.manifest).resolve()
        jobs = json.loads(manifest.read_text())
        if not 0 <= args.index < len(jobs):
            parser.error(f"Index must be in [0, {len(jobs) - 1}]")
        print(run(manifest.parent / jobs[args.index], args.resume))
    elif args.command == "summarize":
        result = json.dumps(summarize(args.chain), indent=2, allow_nan=False) + "\n"
        if args.output:
            Path(args.output).write_text(result)
        else:
            print(result, end="")
    elif args.command == "collect":
        rows = collect(args.directory)
        if not rows:
            parser.error("No sweep configs found")
        fields = list(dict.fromkeys(k for row in rows for k in row))
        with open(args.output, "w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    elif args.command == "extend":
        print(
            json.dumps(
                extend(args.config, args.steps, args.chunk, args.max_chunks), indent=2
            )
        )
    elif args.command == "grid":
        import matplotlib.pyplot as plt
        from .grids import sweep_grid

        fig, _ = sweep_grid(
            args.directory,
            args.x,
            args.y,
            args.kind,
            args.parameters,
            args.models,
            args.time,
            args.truth_models,
        )
        fig.savefig(args.output, dpi=180, bbox_inches="tight")
        plt.close(fig)
    elif args.command == "check-bands":
        from .config import load_config
        from .data import get_bands
        from .physics import band_diagnostics

        cfg = load_config(args.config)
        params = {
            **cfg["FIXED_PARAMS"],
            **cfg.get("TRUE_PARAMS", cfg.get("ALL_PARAMS", cfg["FREE_PARAMS"])),
        }
        result = (
            json.dumps(band_diagnostics(get_bands(cfg["BANDS"]), params), indent=2)
            + "\n"
        )
        if args.output:
            Path(args.output).write_text(result)
        else:
            print(result, end="")
    elif args.command == "compare":
        from .comparison import compare_configs
        import matplotlib.pyplot as plt

        fig = compare_configs(
            args.configs,
            args.kind,
            args.parameters,
            args.orientation,
            args.time,
            args.discard,
        )
        fig.savefig(args.output, dpi=180, bbox_inches="tight")
        plt.close(fig)
    elif args.command == "plot":
        (plot_chain if args.kind == "trace" else plot_corner)(args.chain, args.output)


if __name__ == "__main__":
    main()
