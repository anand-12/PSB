#!/usr/bin/env bash
# Build ../dist/psb_tpu.zip (code, grids, data; no results or caches).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$(dirname "$HERE")/dist"
mkdir -p "$OUT"
rm -f "$OUT/psb_tpu.zip"
cd "$(dirname "$HERE")"
zip -qr "$OUT/psb_tpu.zip" psb_tpu \
    -x 'psb_tpu/results/*' 'psb_tpu/.venv_path' '*/__pycache__/*' '*.pyc' '*/.pytest_cache/*'
echo "wrote $OUT/psb_tpu.zip ($(du -h "$OUT/psb_tpu.zip" | cut -f1))"
