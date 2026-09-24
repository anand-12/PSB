#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Posterior-bridge continual learning: benchmark sweep.
#
#   bash scripts/run_benchmarks.sh              # everything
#   bash scripts/run_benchmarks.sh rotated      # one group
#   bash scripts/run_benchmarks.sh rotated split
#
# Groups: permuted | rotated | split | splitclass
#
# Environment overrides:
#   SEEDS="0 1 2"     seeds to run          (default "0")
#   GPUS="0 1"        visible devices       (default "0 1")
#   PER_GPU=2         concurrent runs / GPU (default 2; rotated runs peak ~4GB)
#   OUTPUT=dir        results directory     (default runs_benchmarks)
#   PYTHON=python     interpreter
#
# Each run is ~2-10 min. A full single-seed sweep is ~20 min on two cards.
# ---------------------------------------------------------------------------
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
GPUS="${GPUS:-0 1}"
PER_GPU="${PER_GPU:-2}"
OUTPUT="${OUTPUT:-runs_benchmarks}"
SEEDS="${SEEDS:-0}"

read -r -a GPU_ARRAY <<< "$GPUS"
NGPU=${#GPU_ARRAY[@]}
POOL=$(( NGPU * PER_GPU ))

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "$OUTPUT/logs"

# Settings shared by every run. These are the values tuned on Permuted MNIST.
COMMON=(
  --particles 128
  --batch-size 64
  --transition-mode bridge
  --anchor-mode particle
  --no-precision-normalize
  --precision-decay 0.8
  --anneal-steps 25
  --fisher-examples 20000
  --bridge-warmup-tasks 99          # reference traversal only; no learned drift
  --data data
)

# tag|extra args
JOBS_PERMUTED=(
  "P_ref|--benchmark permuted --tasks 10 --anchor-strength 4 --first-task-steps 3000 --bridge-reference-moves 100"
)

# Rotation changes the input geometry while the labels stay fixed, so the
# anchor strength that suits Permuted MNIST may not transfer. Sweep it.
JOBS_ROTATED=(
  "R_a02|--benchmark rotated --tasks 10 --anchor-strength 2  --first-task-steps 3000 --bridge-reference-moves 100"
  "R_a04|--benchmark rotated --tasks 10 --anchor-strength 4  --first-task-steps 3000 --bridge-reference-moves 100"
  "R_a08|--benchmark rotated --tasks 10 --anchor-strength 8  --first-task-steps 3000 --bridge-reference-moves 100"
  "R_a16|--benchmark rotated --tasks 10 --anchor-strength 16 --first-task-steps 3000 --bridge-reference-moves 100"
)

# Split MNIST, domain-incremental: 5 binary tasks over a shared two-way head.
# Tasks are ~12.6k examples, so fewer steps than the full-MNIST benchmarks.
JOBS_SPLIT=(
  "S_a01|--benchmark split --split-mode domain --tasks 5 --anchor-strength 1  --first-task-steps 2000 --bridge-reference-moves 40"
  "S_a04|--benchmark split --split-mode domain --tasks 5 --anchor-strength 4  --first-task-steps 2000 --bridge-reference-moves 40"
  "S_a16|--benchmark split --split-mode domain --tasks 5 --anchor-strength 16 --first-task-steps 2000 --bridge-reference-moves 40"
)

# Class-incremental, included to measure the failure rather than assume it:
# only two classes are present per task, so the absent classes' logits are
# never re-calibrated. Expect near-chance. Do not tune this.
JOBS_SPLITCLASS=(
  "S_cls|--benchmark split --split-mode class --tasks 5 --anchor-strength 4 --first-task-steps 2000 --bridge-reference-moves 40"
)

# NB: do not name this GROUPS -- that is a bash special variable holding the
# caller's group IDs, and assignments to it are silently ignored.
SELECTED=("$@")
if [[ ${#SELECTED[@]} -eq 0 ]]; then
  SELECTED=(permuted rotated split splitclass)
fi

JOBS=()
for group in "${SELECTED[@]}"; do
  case "$group" in
    permuted)   JOBS+=("${JOBS_PERMUTED[@]}") ;;
    rotated)    JOBS+=("${JOBS_ROTATED[@]}") ;;
    split)      JOBS+=("${JOBS_SPLIT[@]}") ;;
    splitclass) JOBS+=("${JOBS_SPLITCLASS[@]}") ;;
    *) echo "unknown group: $group (expected permuted|rotated|split|splitclass)"; exit 1 ;;
  esac
done

echo "runs:    $(( ${#JOBS[@]} * $(wc -w <<< "$SEEDS") ))"
echo "gpus:    ${GPU_ARRAY[*]}  (${PER_GPU} concurrent each, pool ${POOL})"
echo "output:  $OUTPUT"
echo

PIDS=()
trap 'echo; echo "interrupted, stopping"; kill ${PIDS[*]} 2>/dev/null; exit 130' INT TERM

index=0
running=0
for seed in $SEEDS; do
  for job in "${JOBS[@]}"; do
    tag="${job%%|*}"
    args="${job#*|}"
    gpu="${GPU_ARRAY[$(( index % NGPU ))]}"
    log="$OUTPUT/logs/${tag}_seed${seed}.log"
    echo "[gpu $gpu] $tag seed $seed"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u -m bwd_cl.train \
      "${COMMON[@]}" $args --seed "$seed" --tag "$tag" --output "$OUTPUT" \
      > "$log" 2>&1 &
    PIDS+=($!)
    index=$(( index + 1 ))
    running=$(( running + 1 ))
    if (( running >= POOL )); then
      wait -n
      running=$(( running - 1 ))
    fi
  done
done
wait

echo
echo "================================ results ================================"
"$PYTHON" - "$OUTPUT" <<'PYEOF'
import json, pathlib, sys, statistics, re, collections

root = pathlib.Path(sys.argv[1])
rows = collections.defaultdict(list)
for metrics in sorted(root.glob("*/metrics.jsonl")):
    records = [json.loads(line) for line in open(metrics)]
    if not records:
        continue
    final = records[-1]
    name = metrics.parent.name
    tag = re.sub(r"_seed\d+$", "", name)
    acquired = [r["task_accuracies"][-1] for r in records]
    expected = max(r["task"] for r in records)
    rows[tag].append({
        "benchmark": final.get("benchmark", "?"),
        "tasks": final["task"],
        "complete": final["task"] == expected == len(records),
        "avg": final["average_accuracy"],
        "forget": final["forgetting"],
        "acq": sum(acquired) / len(acquired),
        "final": final["task_accuracies"],
    })

if not rows:
    print("no completed runs found in", root)
    sys.exit(0)

header = f"{'tag':10s} {'benchmark':10s} {'T':>3s} {'n':>2s} {'avg acc':>16s} {'forget':>8s} {'acquis':>8s}"
print(header)
print("-" * len(header))
incomplete = {t: [r for r in rs if not r["complete"]] for t, rs in rows.items()}
for tag, runs in sorted(rows.items(), key=lambda kv: -statistics.mean(r["avg"] for r in kv[1])):
    runs = [r for r in runs if r["complete"]] or runs
    values = [r["avg"] for r in runs]
    spread = f" +/- {statistics.stdev(values):.4f}" if len(values) > 1 else "        "
    r0 = runs[0]
    print(f"{tag:10s} {r0['benchmark']:10s} {r0['tasks']:3d} {len(runs):2d} "
          f"{statistics.mean(values):8.4f}{spread} "
          f"{statistics.mean(r['forget'] for r in runs):8.4f} "
          f"{statistics.mean(r['acq'] for r in runs):8.4f}")

bad = {t: v for t, v in incomplete.items() if v}
if bad:
    print("\nWARNING - runs that did not finish, excluded from the means above:")
    for tag, v in sorted(bad.items()):
        print(f"  {tag}: {len(v)} run(s) stopped early "
              f"(tasks reached: {sorted(r['tasks'] for r in v)})")

print("\nper-task accuracy after the full stream (seed 0):")
for tag, runs in sorted(rows.items()):
    print(f"  {tag:10s} " + " ".join(f"{a:.3f}" for a in runs[0]["final"]))
PYEOF
