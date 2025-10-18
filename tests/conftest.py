import os
import sys

# Avoid OpenMP runtime duplication aborts on macOS with mixed Python wheels
# (e.g., PyTorch, NumPy, and other libraries linking different libomp).
# This setting is an unsafe workaround but acceptable for unit tests.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Keep thread count small to reduce contention during tests.
os.environ.setdefault("OMP_NUM_THREADS", "1")

# Ensure repository root is in sys.path for absolute imports like `import src.*`
REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)