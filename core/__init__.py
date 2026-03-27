"""Legacy compatibility imports for repo-local execution."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from .astrometry import AstrometrySolver
from .photometry import Photometry
from .target import TargetManager

__all__ = ["AstrometrySolver", "Photometry", "TargetManager"]
