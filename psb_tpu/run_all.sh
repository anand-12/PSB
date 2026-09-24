#!/usr/bin/env bash
# Run every phase in order. Safe to rerun at any time: finished runs are
# skipped and unfinished ones resume from their last checkpoint.
#
# Environment knobs (all optional):
#   PHASES="phase0_parity phase1a_tune"   subset / order of grids/<name>.yaml
#   COST_CAP_USD=150        stop once chips x PRICE_PER_CHIP_HOUR x elapsed exceeds this
#   PRICE_PER_CHIP_HOUR=1.35   v6e Flex-start (v5p 2.10, v5e 0.60)
#   GCS_BUCKET=my-bucket    mirror results to gs://my-bucket/psb_results after each phase
#   SELF_DELETE=1           delete this VM when finished (results stay on the data disk)
#   FORCE=1                 continue even if the phase-0 parity gate fails
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
DATA_ROOT="${DATA_ROOT:-/mnt/data}"
mountpoint -q "$DATA_ROOT" 2>/dev/null || [ "$DATA_ROOT" != /mnt/data ] || DATA_ROOT="$HOME/psb_data"
VENV="${VENV:-$(cat "$HERE/.venv_path" 2>/dev/null || echo "$DATA_ROOT/venv")}"
PY="$VENV/bin/python"
[ -x "$PY" ] || { echo "no Python at $PY -- run setup.sh first"; exit 1; }

export PSB_RESULTS="${PSB_RESULTS:-$DATA_ROOT/psb_results}"
export PSB_JAX_CACHE="${PSB_JAX_CACHE:-$DATA_ROOT/jax_cache}"
PHASES="${PHASES:-smoke phase0_parity phase1a_tune phase1b_seeds phase2a_schedules phase2b_budget phase3_streams phase4_backlog}"
COST_CAP_USD="${COST_CAP_USD:-150}"
PRICE_PER_CHIP_HOUR="${PRICE_PER_CHIP_HOUR:-1.35}"
LOGS="$PSB_RESULTS/logs"
mkdir -p "$LOGS" "$PSB_JAX_CACHE"
rm -f "$PSB_RESULTS/COST_CAP_REACHED"

CHIPS="$("$PY" -c 'import jax; print(jax.device_count())' 2>/dev/null)"
CHIPS="${CHIPS:-1}"
START="$(date +%s)"
spent() { awk -v s="$(( $(date +%s) - START ))" -v c="$CHIPS" -v p="$PRICE_PER_CHIP_HOUR" \
          'BEGIN { printf "%.2f", s / 3600 * c * p }'; }

sync_results() {
    [ -n "${GCS_BUCKET:-}" ] || return 0
    command -v gcloud >/dev/null || { echo "warning: gcloud missing; skipping GCS sync"; return 0; }
    gcloud storage rsync -r -x '.*/_state/.*' "$PSB_RESULTS" "gs://$GCS_BUCKET/psb_results" \
        >/dev/null 2>&1 || echo "warning: GCS sync failed"
}

# Watchdog: stop the sweep if the estimated spend passes the cap.
(
    while sleep 60; do
        if awk -v a="$(spent)" -v b="$COST_CAP_USD" 'BEGIN { exit !(a > b) }'; then
            echo "[run_all] estimated spend \$$(spent) > cap \$$COST_CAP_USD; stopping" | tee -a "$LOGS/run_all.log"
            touch "$PSB_RESULTS/COST_CAP_REACHED"
            pkill -f "psb.run" || true
            exit 0
        fi
    done
) &
WATCHDOG=$!
trap 'kill $WATCHDOG 2>/dev/null' EXIT

echo "[run_all] $CHIPS chip(s), cap \$$COST_CAP_USD at \$$PRICE_PER_CHIP_HOUR/chip-h; results in $PSB_RESULTS" | tee -a "$LOGS/run_all.log"
for phase in $PHASES; do
    [ -f "$PSB_RESULTS/COST_CAP_REACHED" ] && break
    grid="grids/$phase.yaml"
    [ -f "$grid" ] || { echo "no grid $grid"; exit 1; }
    echo "[run_all] === $phase ($(date)) spent so far ~\$$(spent) ===" | tee -a "$LOGS/run_all.log"
    "$PY" -m psb.run "$grid" 2>&1 | tee -a "$LOGS/$phase.log"
    status="${PIPESTATUS[0]}"
    "$PY" -m psb.summarize "$PSB_RESULTS/$phase" > /dev/null 2>&1 || true
    sync_results
    [ -f "$PSB_RESULTS/COST_CAP_REACHED" ] && break
    if [ "$status" != 0 ]; then
        echo "[run_all] $phase failed (exit $status); see $LOGS/$phase.log. Rerun to resume." | tee -a "$LOGS/run_all.log"
        exit "$status"
    fi
    if [ "$phase" = phase0_parity ]; then
        if ! "$PY" -m psb.gate "$grid" --results "$PSB_RESULTS" | tee -a "$LOGS/run_all.log"; then
            if [ "${FORCE:-0}" != 1 ]; then
                echo "[run_all] parity gate FAILED; stopping. Inspect, then FORCE=1 to continue." | tee -a "$LOGS/run_all.log"
                exit 1
            fi
        fi
    fi
done

echo "[run_all] finished $(date); estimated spend ~\$$(spent)" | tee -a "$LOGS/run_all.log"
sync_results
if [ "${SELF_DELETE:-0}" = 1 ] && command -v gcloud >/dev/null; then
    meta="http://metadata.google.internal/computeMetadata/v1/instance"
    name="$(curl -s -H 'Metadata-Flavor: Google' "$meta/name")"
    zone="$(curl -s -H 'Metadata-Flavor: Google' "$meta/zone" | awk -F/ '{print $NF}')"
    echo "[run_all] deleting VM $name in $zone (data disk is kept)" | tee -a "$LOGS/run_all.log"
    gcloud compute instances delete "$name" --zone "$zone" --quiet
fi
