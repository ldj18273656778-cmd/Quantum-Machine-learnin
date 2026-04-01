from __future__ import annotations

from pathlib import Path

import numpy as np

# ======= 手动修改这里的参数 =======
N1 = 3
M = 4
SEED = 117
ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = ROOT / "code" / "Train" / "data" / "theta_demo.npy"
# ================================

rng = np.random.default_rng(SEED)
theta = rng.uniform(0.0, 2 * np.pi, size=(N1, M)).astype(float)

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
np.save(OUT_PATH, theta)
print(f"saved: {OUT_PATH}")
print(f"shape: {theta.shape}")
print(f"root: {ROOT}")
