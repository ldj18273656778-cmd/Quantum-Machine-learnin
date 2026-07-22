"""JAX runtime configuration for the isolated parity backend."""

from __future__ import annotations

import os

_ = os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax

jax.config.update("jax_enable_x64", True)

__all__ = ["jax"]
