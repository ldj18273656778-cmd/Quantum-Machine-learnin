"""Compatibility shim so scripts under code/Train can `import sampling` directly."""

from pathlib import Path
from importlib import import_module
import pkgutil

__path__ = pkgutil.extend_path(__path__, __name__)
_real_sampling = Path(__file__).resolve().parents[2] / "sampling"
if _real_sampling.exists():
    __path__.append(str(_real_sampling))

from .DQNN_generate_y import DQNN_generate_y
from .ISQNN_generate_y import ISQNN_generate_y

__all__ = ["DQNN_generate_y", "ISQNN_generate_y"]


def __getattr__(name):
    if name == "generate_probability_distribution":
        return import_module(".probility_distribution", __name__).generate_probability_distribution
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
