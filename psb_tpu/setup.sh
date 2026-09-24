#!/usr/bin/env bash
# One-time environment setup on a TPU VM (safe to rerun).
#
# Ubuntu 22.04 ships Python 3.10, but JAX 0.10 needs >= 3.11, so this installs
# uv and lets it fetch Python 3.12. The venv lives on the persistent data disk,
# so a new VM with the same disk skips the install.
#
#   bash setup.sh                         # TPU VM (default: jax[tpu]==0.10.2)
#   JAX_SPEC="jax[cuda12]" bash setup.sh  # a GPU box instead
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
DATA_ROOT="${DATA_ROOT:-/mnt/data}"
if ! mountpoint -q "$DATA_ROOT" 2>/dev/null && [ "$DATA_ROOT" = /mnt/data ]; then
    echo "warning: /mnt/data is not a mounted disk; using \$HOME/psb_data (lost when the VM is deleted)"
    DATA_ROOT="$HOME/psb_data"
fi
mkdir -p "$DATA_ROOT"
VENV="${VENV:-$DATA_ROOT/venv}"
JAX_SPEC="${JAX_SPEC:-jax[tpu]==0.10.2}"

if command -v apt-get >/dev/null; then
    sudo apt-get update -qq && sudo apt-get install -y -qq unzip tmux curl >/dev/null
fi
if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
    export PATH="$HOME/.local/bin:$PATH"
fi
# Keep uv's Python on the data disk too: a venv on the persistent disk that
# points at an interpreter on a deleted VM's boot disk is broken.
export UV_PYTHON_INSTALL_DIR="$DATA_ROOT/uv-python"
if ! "$VENV/bin/python" -c "import sys" >/dev/null 2>&1; then
    uv venv --quiet --clear --python 3.12 "$VENV"
fi
uv pip install --quiet --python "$VENV/bin/python" "$JAX_SPEC" \
    -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
uv pip install --quiet --python "$VENV/bin/python" -r "$HERE/requirements.txt"

"$VENV/bin/python" -c "import jax; d = jax.devices(); print(f'JAX {jax.__version__}: {len(d)} x {d[0].device_kind} ({d[0].platform})')"
(cd "$HERE" && JAX_PLATFORMS=cpu "$VENV/bin/python" -m pytest -q tests)
echo "$VENV" > "$HERE/.venv_path"
echo
echo "Setup OK (venv: $VENV). Start the sweep inside tmux:"
echo "  tmux new -s psb 'bash $HERE/run_all.sh'"
