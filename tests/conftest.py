import sys
from pathlib import Path

# Ensure root package and neural_context are importable
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
