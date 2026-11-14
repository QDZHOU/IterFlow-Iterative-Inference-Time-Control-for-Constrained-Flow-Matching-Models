"""Maze2D evaluation: native HardFlow and IterFlow."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from hardflow.models_flow.iterflow.io import write_json

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ("checkpoint", "dynamics")

def load(path):
    return json.loads(Path(path).read_text())

def execution_env(threads=4):
    env = os.environ.copy()
    env.update(HARDFLOW_NATIVE_MUJOCO="1", HARDFLOW_TORCH_THREADS=str(threads),
               OMP_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS=str(threads),
               MKL_NUM_THREADS=str(threads), PYTHONDONTWRITEBYTECODE="1")
    env.setdefault("D4RL_DATASET_DIR", str(ROOT / "datasets"))
    return env

def validate_settings(settings):
    # Validate the supported public interface before loading large assets.
    hf = settings["hardflow"]
    if hf["env"] != "maze2d-large-v1" or hf["device"] != "cpu":
        raise ValueError("This release supports maze2d-large-v1 on CPU")
    if hf["guidance_method"] != "hardflow_new" or hf["warmstart_batch"] != 1:
        raise ValueError("IterFlow preserves hardflow_new with warmstart_batch=1")
    if hf["hardflow_activation"] != "late":
        raise ValueError("IterFlow's frozen uncontrolled prefix requires late activation")
    n = hf["ode_t_steps"]
    if not isinstance(n, int) or not 1 <= settings["tail_steps"] < n:
        raise ValueError("tail_steps must be between 1 and ode_t_steps - 1")
    import math
    if len(settings["observation"]) != 4 or len(settings["goal"]) != 2:
        raise ValueError("Expected [x,y,vx,vy] observation and [x,y] goal")
    if not all(math.isfinite(v) for v in settings["observation"] + settings["goal"]):
        raise ValueError("Non-finite endpoints")
    if settings["cost"] != {"kind": "hf", "coefficient": 1.0}:
        raise ValueError("Native HardFlow surrogate coefficient must remain 1")

def asset_report(settings):
    paths = {key: ROOT / settings[key] for key in ASSETS}
    paths["dataset"] = Path(execution_env()["D4RL_DATASET_DIR"]) / "maze2d-large-sparse-v1.hdf5"
    return {k: {"path": str(p), "exists": p.is_file()} for k, p in paths.items()}

def invoke(module, args, env):
    subprocess.run([sys.executable, "-m", module, *map(str, args)], cwd=ROOT, env=env, check=True)

def search_command(manifest, resume=False):
    out = Path(manifest["output"])
    s = manifest["search"]
    args = ["--source", out / "baseline", "--output", out / "run", "--rounds", s["rounds"],
            "--seed", manifest["search_seed"], "--temperature", s["temperature"], "--walk-sigma", s["walk_sigma"]]
    module = "hardflow.models_flow.iterflow.history_ball_noise_hybrid"
    args += ["--source-rounds", 0, "--require-improvement", "--update-balls",
             "--spread-between-samples", "--entry-screen", "--prefer-far-from-center"]
    for key in ("noise_radius", "noise_max_radius", "pool_size", "epsilon_scale", "mcmc_start"):
        args += ["--" + key.replace("_", "-"), s[key]]
    if s["control_norm_ratio"] is not None:
        args += ["--control-norm-ratio", s["control_norm_ratio"]]
    if resume:
        args.append("--resume")
    return module, args

def run(args):
    settings = load(args.config)
    validate_settings(settings)
    search = load(ROOT / "configs/search.json")
    for key in search:
        value = getattr(args, key, None)
        if value is not None:
            search[key] = value
    if args.no_control_cap:
        search["control_norm_ratio"] = None
    if search["rounds"] < 1 or args.threads < 1:
        raise ValueError("rounds and threads must be positive")
    manifest = dict(method=args.method, settings=settings, search=search, seed=args.seed,
                    search_seed=args.search_seed, threads=args.threads, output=str(args.output.resolve()))
    module, command = search_command(manifest)
    if args.dry_run:
        print(json.dumps(dict(manifest=manifest, module=module, arguments=list(map(str, command)), assets=asset_report(settings)), indent=2))
        return
    missing = [k for k,v in asset_report(settings).items() if not v["exists"]]
    if missing:
        raise FileNotFoundError("Missing assets: " + ", ".join(missing) + ". Place the model, dynamics and dataset files at the configured paths first.")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "manifest.json", manifest)
    write_json(out / "settings.json", settings)
    env = execution_env(args.threads)
    invoke("hardflow.models_flow.iterflow.initialization", ["--config", out / "settings.json", "--seed", args.seed, "--output", out / "baseline"], env)
    if args.method == "hardflow":
        return
    if args.method == "iterflow":
        (out / "run").mkdir()
        write_json(out / "run/search_policy.json", search["search_policy"])
    invoke(module, command, env)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="Generate paired baselines and run IterFlow ")
    r.add_argument("--config", type=Path, default=ROOT / "configs/maze2d.json")
    r.add_argument("--output", type=Path, required=True)
    r.add_argument("--method", choices=["iterflow", "hardflow"], default="iterflow")
    r.add_argument("--seed", type=int, default=1051813958)
    r.add_argument("--search-seed", type=int, default=20261113)
    r.add_argument("--threads", type=int, default=4)
    for name, kind in [("rounds",int),("mcmc-start",int),("temperature",float),("walk-sigma",float),
                       ("noise-radius",float),("noise-max-radius",float),("pool-size",int),
                       ("epsilon-scale",float),("control-norm-ratio",float)]:
        r.add_argument("--" + name, type=kind)
    r.add_argument("--no-control-cap", action="store_true")
    r.add_argument("--dry-run", action="store_true")
    resume = sub.add_parser("resume", help="Extend an existing experiment to a new total round count")
    resume.add_argument("experiment", type=Path)
    resume.add_argument("--rounds", type=int, required=True)
    status = sub.add_parser("status", help="Read the latest saved progress")
    status.add_argument("experiment", type=Path)
    doctor = sub.add_parser("doctor", help="Check dependencies and local model/data assets")
    doctor.add_argument("--config", type=Path, default=ROOT / "configs/maze2d.json")
    args = p.parse_args()
    if args.command == "run":
        run(args)
    elif args.command == "doctor":
        modules = ("numpy", "torch", "casadi", "mujoco", "gym", "h5py", "einops", "tyro", "scipy", "yaml", "tqdm")
        deps = {name: importlib.util.find_spec(name) is not None for name in modules}
        assets = asset_report(load(args.config))
        print(json.dumps(dict(dependencies=deps, assets=assets, root=str(ROOT)), indent=2))
        if not all(deps.values()) or not all(v["exists"] for v in assets.values()):
            raise SystemExit(1)
    elif args.command == "status":
        out = args.experiment.resolve()
        file = out / "run/progress.json"
        if not file.exists():
            file = out / "baseline/summary.json"
        print(json.dumps(load(file), indent=2))
    else:
        out = args.experiment.resolve()
        manifest = load(out / "manifest.json")
        if manifest["method"] == "hardflow":
            raise ValueError("Baseline-only experiment has no search to resume")
        completed = load(out / "run/progress.json")["rounds_completed"]
        if args.rounds <= max(completed, manifest["search"]["rounds"]):
            raise ValueError("New total must exceed both completed and previously requested rounds")
        manifest["search"]["rounds"] = args.rounds
        manifest["output"] = str(out)
        module, command = search_command(manifest, resume=True)
        # Keep the old launch manifest if the engine refuses the extension.
        invoke(module, command, execution_env(manifest["threads"]))
        write_json(out / "manifest.json", manifest)

if __name__ == "__main__":
    main()
