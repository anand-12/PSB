"""Run every pending run in a grid file.

    python -m psb.run grids/phase1b_seeds.yaml            # run (resumes automatically)
    python -m psb.run grids/phase1b_seeds.yaml --plan     # show groups and sizes only

Runs whose ``summary.json`` is complete are skipped, so re-running a grid after
a preemption continues where it stopped.
"""
from __future__ import annotations

import argparse
import os
import shutil
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("grid")
    parser.add_argument("--results", default=os.environ.get("PSB_RESULTS", "results"))
    parser.add_argument("--plan", action="store_true", help="print the execution plan and exit")
    parser.add_argument("--only", default=None, help="only run tags containing this substring")
    parser.add_argument("--max-runs-per-device", type=int,
                        default=int(os.environ.get("PSB_MAX_RUNS_PER_DEVICE", "64")))
    args = parser.parse_args()

    import warnings
    warnings.filterwarnings("ignore", message="Some donated buffers were not usable")
    import jax
    jax.config.update("jax_default_matmul_precision", os.environ.get("PSB_MATMUL_PRECISION", "highest"))
    cache = os.environ.get("PSB_JAX_CACHE")
    if cache:
        jax.config.update("jax_compilation_cache_dir", cache)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 2.0)

    from .config import expand, load_grid, static_signature
    from .engine import Engine, is_complete

    grid = load_grid(args.grid)
    root = Path(args.results)
    runs = expand(grid, root)
    if args.only:
        runs = [r for r in runs if args.only in r["tag"]]
    pending = [r for r in runs if not is_complete(root, r)]
    (root / grid["name"]).mkdir(parents=True, exist_ok=True)
    shutil.copy(args.grid, root / grid["name"] / "grid.yaml")

    groups: dict = {}
    for run in pending:
        groups.setdefault(static_signature(run), []).append(run)

    engine = Engine(root, args.max_runs_per_device)
    print(f"[{grid['name']}] {len(runs)} runs, {len(runs) - len(pending)} complete, "
          f"{len(pending)} pending in {len(groups)} shape groups on {engine.devices} "
          f"{jax.devices()[0].platform} device(s)", flush=True)

    equivalents = 0.0
    plan = []
    for signature, members in groups.items():
        static = dict(signature)
        per_device = engine.runs_per_device(static)
        size = per_device * engine.devices
        chunks = [members[i:i + size] for i in range(0, len(members), size)]
        moves = static["first_task_steps"] + (static["tasks"] - 1) * static["stages"] * static["moves_per_stage"]
        weight = moves / (3000 + 9 * 2500) * static["particles"] / 128
        equivalents += weight * len(members)
        plan.append((static, members, chunks))
        print(f"  {static['benchmark']} T={static['tasks']} K={static['particles']} "
              f"{static['method']} M={static['stages']} n={static['moves_per_stage']}: "
              f"{len(members)} runs, {per_device}/device, {len(chunks)} chunk(s)", flush=True)
    print(f"[{grid['name']}] workload ~{equivalents:.0f} base-run equivalents "
          f"(1 = 10-task Permuted MNIST, K=128, 2,500 moves/boundary)", flush=True)
    if args.plan:
        return

    started = time.time()
    for g, (static, members, chunks) in enumerate(plan):
        for c, chunk in enumerate(chunks):
            label = f"{grid['name']} g{g + 1}/{len(plan)} c{c + 1}/{len(chunks)}"
            engine.run_chunk(chunk, static, label)
    print(f"[{grid['name']}] done in {(time.time() - started) / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
