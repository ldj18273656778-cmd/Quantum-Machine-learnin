"""Estimate theta_j from filtered samples, accounting for Rx(phi) on neighbors.

The estimation formula depends on the number of CZ neighbors d:

    P(y_j=1 | x_j=0, x_neighbors=1) = (1 + alpha_d * cos(theta_j)) / 2

where alpha_d = -(cos phi)^d  for Rx(phi) on each neighbor.

For phi=pi/4:  alpha_d = -(1/sqrt(2))^d

For the original circuit (phi=0): alpha_d = -1 for all d,
recovering the classic formula theta_hat = arccos(1 - 2*E[y]).
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from Train.find_x_indices_by_graph_condition import find_indices


def estimate_theta_rx(
    x: np.ndarray,
    y: np.ndarray,
    target_bit: int,
    adjacency: list[set[int]],
    rx_angle: float = np.pi / 4,
    show_progress: bool = False,
) -> dict:
    """Estimate theta_j from filtered samples with Rx-modification.

    Parameters
    ----------
    x : np.ndarray, shape (N,), dtype str
        Bitstring inputs, each of length n.
    y : np.ndarray, shape (N, n), dtype int
        Target outputs.
    target_bit : int
        Index of the target bit (0-based).
    adjacency : list[set[int]]
        Adjacency list, adjacency[j] = set of neighbor indices.
    rx_angle : float
        Rx rotation angle in radians. Default pi/4.
    show_progress : bool
        Show progress bar during filtering.

    Returns
    -------
    dict with keys:
        target_bit_0based, N_sp, sum_y, theta_hat_rad, theta_hat_deg,
        alpha_d, n_neighbors, indices_0based
    """
    indices = find_indices(
        x=x, target_bit=target_bit, adjacency=adjacency, show_progress=show_progress
    )

    n_sp = len(indices)
    if n_sp == 0:
        raise ValueError(
            f"N_sp = 0 for target_bit={target_bit}, cannot estimate theta."
        )

    if not (0 <= target_bit < y.shape[1]):
        raise ValueError(
            f"target_bit={target_bit} out of range, y.shape[1]={y.shape[1]}"
        )

    # Number of CZ neighbors
    d = len(adjacency[target_bit])

    # alpha_d = -(cos phi)^d
    alpha_d = -float(np.cos(rx_angle)) ** d

    y_selected = y[indices, target_bit]
    if not np.isin(y_selected, [0, 1]).all():
        raise ValueError("y_selected must be binary {0,1}.")

    y_sum = int(np.sum(y_selected))
    p_hat = y_sum / n_sp  # P(y=1)

    # theta_hat = arccos((2P - 1) / alpha_d)
    arg = (2.0 * p_hat - 1.0) / alpha_d
    arg = float(np.clip(arg, -1.0, 1.0))
    theta_hat = float(np.arccos(arg))

    return {
        "target_bit_0based": int(target_bit),
        "N_sp": n_sp,
        "sum_y": y_sum,
        "p_hat": float(p_hat),
        "alpha_d": alpha_d,
        "n_neighbors": d,
        "rx_angle": float(rx_angle),
        "theta_hat_rad": theta_hat,
        "theta_hat_deg": float(np.degrees(theta_hat)),
        "indices_0based": indices,
    }
