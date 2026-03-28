from .DQNN_generate_y import DQNN_generate_y
from .ISQNN_generate_y import ISQNN_generate_y
from importlib import import_module

__all__ = [
	"DQNN_generate_y",
	"ISQNN_generate_y",
]


def __getattr__(name):
	if name == "generate_probability_distribution":
		mod = import_module(".probility_distribution", __name__)
		return mod.generate_probability_distribution
	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
