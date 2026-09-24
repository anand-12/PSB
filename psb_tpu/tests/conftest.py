import os
import sys
from pathlib import Path

# A TPU can be held by one process only; tests never compete with a sweep.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
