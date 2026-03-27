"""Legacy compatibility imports for repo-local execution."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from .astrometry import AstrometrySolver  # noqa: E402
from .photometry import Photometry  # noqa: E402
from .target import TargetManager  # noqa: E402

__all__ = ["AstrometrySolver", "Photometry", "TargetManager"]
