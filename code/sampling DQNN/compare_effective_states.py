"""
Layer-wise effective-state comparison for DQNN vs ISQNN.

Definitions used in this script:
- DQNN pre_local: the m-qubit state right after one layer's readout/reset logic.
- DQNN post_local: the state after additionally applying that layer's Rz/CZ block.
- ISQNN pre_local: the reduced density matrix of the current slice, conditioned on
  the same past readout history, with that slice's own local Rz/CZ omitted.
- ISQNN post_local: the same reduced state after restoring that slice's local Rz/CZ.

The comparison is done in density-matrix form, so it remains meaningful even when
the ISQNN current slice is mixed because it is still entangled with future slices.
"""

import argparse
import json
import math
import random
from dataclasses import dataclass

import numpy as np


H_GATE = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128) / math.sqrt(2.0)
X_GATE = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
Z_GATE = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
CZ_GATE = np.diag([1.0, 1.0, 1.0, -1.0]).astype(np.complex128)
EPS = 1e-12


@dataclass
class DQNNLayerTrace:
    layer_index: int
    readout: list[int]
    pre_local_state: np.ndarray
    post_local_state: np.ndarray


def rz_gate(theta: float) -> np.ndarray:
    half = theta / 2.0
    return np.array(
        [
            [np.exp(-1j * half), 0.0],
            [0.0, np.exp(1j * half)],
        ],
        dtype=np.complex128,
    )


def num_qubits_from_state(state: np.ndarray) -> int:
    size = int(state.size)
    num_qubits = int(round(math.log2(size)))
    if (1 << num_qubits) != size:
        raise ValueError(f"State vector size {size} is not a power of two.")
    return num_qubits


def normalize_state(state: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(state)
    if norm <= EPS:
        raise ValueError("Encountered a zero-norm state.")
    return state / norm


def zero_state(num_qubits: int) -> np.ndarray:
    state = np.zeros(1 << num_qubits, dtype=np.complex128)
    state[0] = 1.0
    return state


def apply_single_qubit_gate(
    state: np.ndarray,
    gate: np.ndarray,
    qubit: int,
    num_qubits: int | None = None,
) -> np.ndarray:
    if num_qubits is None:
        num_qubits = num_qubits_from_state(state)
    tensor = state.reshape((2,) * num_qubits)
    tensor = np.moveaxis(tensor, qubit, 0)
    tensor = np.tensordot(gate, tensor, axes=([1], [0]))
    tensor = np.moveaxis(tensor, 0, qubit)
    return tensor.reshape(-1)


def apply_two_qubit_gate(
    state: np.ndarray,
    gate: np.ndarray,
    qubit_a: int,
    qubit_b: int,
    num_qubits: int | None = None,
) -> np.ndarray:
    if qubit_a == qubit_b:
        raise ValueError("The two target qubits must be different.")
    if qubit_a > qubit_b:
        qubit_a, qubit_b = qubit_b, qubit_a
    if num_qubits is None:
        num_qubits = num_qubits_from_state(state)
    gate_tensor = gate.reshape(2, 2, 2, 2)
    tensor = state.reshape((2,) * num_qubits)
    tensor = np.moveaxis(tensor, [qubit_a, qubit_b], [0, 1])
    tensor = np.tensordot(gate_tensor, tensor, axes=([2, 3], [0, 1]))
    tensor = np.moveaxis(tensor, [0, 1], [qubit_a, qubit_b])
    return tensor.reshape(-1)


def z_outcome_probabilities(
    state: np.ndarray,
    qubit: int,
    num_qubits: int | None = None,
) -> tuple[float, float]:
    if num_qubits is None:
        num_qubits = num_qubits_from_state(state)
    tensor = state.reshape((2,) * num_qubits)
    tensor = np.moveaxis(tensor, qubit, 0)
    prob0 = float(np.vdot(tensor[0].reshape(-1), tensor[0].reshape(-1)).real)
    prob1 = float(np.vdot(tensor[1].reshape(-1), tensor[1].reshape(-1)).real)
    return prob0, prob1


def project_z(
    state: np.ndarray,
    qubit: int,
    outcome: int,
    num_qubits: int | None = None,
) -> tuple[np.ndarray, float]:
    if num_qubits is None:
        num_qubits = num_qubits_from_state(state)
    tensor = state.reshape((2,) * num_qubits).copy()
    tensor = np.moveaxis(tensor, qubit, 0)
    kept = tensor[outcome].copy()
    probability = float(np.vdot(kept.reshape(-1), kept.reshape(-1)).real)
    if probability <= EPS:
        raise ValueError(
            f"Forced outcome {outcome} on qubit {qubit} has zero probability."
        )
    tensor[1 - outcome] = 0.0
    tensor[outcome] /= math.sqrt(probability)
    tensor = np.moveaxis(tensor, 0, qubit)
    return tensor.reshape(-1), probability


def measure_x_with_forced_outcome(
    state: np.ndarray,
    qubit: int,
    outcome: int,
    num_qubits: int | None = None,
) -> tuple[np.ndarray, float]:
    if num_qubits is None:
        num_qubits = num_qubits_from_state(state)
    rotated = apply_single_qubit_gate(state, H_GATE, qubit, num_qubits)
    return project_z(rotated, qubit, outcome, num_qubits)


def reset_measured_qubit(
    state: np.ndarray,
    qubit: int,
    outcome: int,
    num_qubits: int | None = None,
) -> np.ndarray:
    if num_qubits is None:
        num_qubits = num_qubits_from_state(state)
    if outcome == 1:
        return apply_single_qubit_gate(state, X_GATE, qubit, num_qubits)
    return state


def reduced_density_matrix(
    state: np.ndarray,
    keep_qubits: list[int],
    num_qubits: int | None = None,
) -> np.ndarray:
    if num_qubits is None:
        num_qubits = num_qubits_from_state(state)
    keep_qubits = list(keep_qubits)
    trace_qubits = [q for q in range(num_qubits) if q not in keep_qubits]
    tensor = state.reshape((2,) * num_qubits)
    tensor = np.transpose(tensor, keep_qubits + trace_qubits)
    dim_keep = 1 << len(keep_qubits)
    dim_trace = 1 << len(trace_qubits)
    matrix = tensor.reshape(dim_keep, dim_trace)
    return matrix @ matrix.conj().T


def density_from_state(state: np.ndarray) -> np.ndarray:
    state = normalize_state(state)
    return np.outer(state, state.conj())


def density_purity(rho: np.ndarray) -> float:
    return float(np.trace(rho @ rho).real)


def frobenius_distance_to_pure_state(psi: np.ndarray, rho: np.ndarray) -> float:
    target = density_from_state(psi)
    return float(np.linalg.norm(target - rho))


def pure_state_fidelity(psi: np.ndarray, rho: np.ndarray) -> float:
    psi = normalize_state(psi)
    value = np.vdot(psi, rho @ psi)
    return float(np.clip(value.real, 0.0, 1.0))


def normalize_theta_list(theta_list, total_length: int) -> list[float]:
    if isinstance(theta_list, (int, float)):
        return [float(theta_list)] * total_length

    values = [float(value) for value in theta_list]
    if not values:
        raise ValueError("theta_list must not be empty.")
    if len(values) < total_length:
        values.extend([values[-1]] * (total_length - len(values)))
    elif len(values) > total_length:
        values = values[:total_length]
    return values


def apply_slice_local_ops_to_state(
    state: np.ndarray,
    theta_list: list[float],
    slice_index: int,
    m: int,
) -> np.ndarray:
    state = state.copy()
    for qubit in range(m):
        theta_idx = slice_index * m + qubit
        state = apply_single_qubit_gate(state, rz_gate(theta_list[theta_idx]), qubit, m)
    for pair in range(m // 2):
        state = apply_two_qubit_gate(state, CZ_GATE, 2 * pair, 2 * pair + 1, m)
    return normalize_state(state)


def slice_local_unitary(theta_list: list[float], slice_index: int, m: int) -> np.ndarray:
    dim = 1 << m
    basis = np.eye(dim, dtype=np.complex128)
    columns = [
        apply_slice_local_ops_to_state(basis[:, col], theta_list, slice_index, m)
        for col in range(dim)
    ]
    return np.column_stack(columns)


def prepare_slice_input_state(bitstring: str, slice_index: int, m: int) -> np.ndarray:
    state = zero_state(m)
    for qubit in range(m):
        if bitstring[slice_index * m + qubit] == "0":
            state = apply_single_qubit_gate(state, H_GATE, qubit, m)
    return normalize_state(state)


def trace_dqnn_layers(
    bitstring: str,
    n1: int,
    m: int,
    theta_list: list[float],
    trajectory_seed: int,
) -> tuple[np.ndarray, list[DQNNLayerTrace], list[int]]:
    rng = random.Random(trajectory_seed)
    state = prepare_slice_input_state(bitstring, 0, m)
    state = apply_slice_local_ops_to_state(state, theta_list, 0, m)
    layer_traces: list[DQNNLayerTrace] = []

    for layer_index in range(1, n1):
        readout: list[int] = []
        for qubit in range(m):
            x_bit = bitstring[layer_index * m + qubit]
            if x_bit == "0":
                state = apply_single_qubit_gate(state, H_GATE, qubit, m)
                outcome = rng.randint(0, 1)
                readout.append(outcome)
                if outcome == 1:
                    state = apply_single_qubit_gate(state, X_GATE, qubit, m)
            else:
                rotated = apply_single_qubit_gate(state, H_GATE, qubit, m)
                prob0, prob1 = z_outcome_probabilities(rotated, qubit, m)
                threshold = rng.random()
                outcome = 0 if threshold < prob0 / max(prob0 + prob1, EPS) else 1
                readout.append(outcome)
                collapsed, _ = project_z(rotated, qubit, outcome, m)
                state = reset_measured_qubit(collapsed, qubit, outcome, m)

        pre_local_state = normalize_state(state.copy())
        post_local_state = apply_slice_local_ops_to_state(pre_local_state, theta_list, layer_index, m)
        state = post_local_state
        layer_traces.append(
            DQNNLayerTrace(
                layer_index=layer_index,
                readout=readout,
                pre_local_state=pre_local_state,
                post_local_state=post_local_state,
            )
        )

    final_x_readout: list[int] = []
    final_state = state.copy()
    for qubit in range(m):
        rotated = apply_single_qubit_gate(final_state, H_GATE, qubit, m)
        prob0, prob1 = z_outcome_probabilities(rotated, qubit, m)
        threshold = rng.random()
        outcome = 0 if threshold < prob0 / max(prob0 + prob1, EPS) else 1
        final_x_readout.append(outcome)
        final_state, _ = project_z(rotated, qubit, outcome, m)

    return state, layer_traces, final_x_readout


def build_isqnn_pre_measurement_state(
    bitstring: str,
    n1: int,
    m: int,
    theta_list: list[float],
    omit_local_slice: int | None,
) -> np.ndarray:
    total_qubits = n1 * m
    state = zero_state(total_qubits)

    for qubit in range(total_qubits):
        if bitstring[qubit] == "0":
            state = apply_single_qubit_gate(state, H_GATE, qubit, total_qubits)

    for slice_index in range(n1):
        if omit_local_slice is not None and slice_index == omit_local_slice:
            continue
        for offset in range(m):
            qubit = slice_index * m + offset
            theta_idx = slice_index * m + offset
            state = apply_single_qubit_gate(
                state,
                rz_gate(theta_list[theta_idx]),
                qubit,
                total_qubits,
            )
        for pair in range(m // 2):
            left = slice_index * m + 2 * pair
            right = left + 1
            state = apply_two_qubit_gate(state, CZ_GATE, left, right, total_qubits)

    for slice_index in range(n1 - 1):
        for offset in range(m):
            current_qubit = slice_index * m + offset
            next_qubit = (slice_index + 1) * m + offset
            state = apply_two_qubit_gate(
                state,
                CZ_GATE,
                current_qubit,
                next_qubit,
                total_qubits,
            )

    return normalize_state(state)


def conditional_isqnn_slice_density(
    bitstring: str,
    n1: int,
    m: int,
    theta_list: list[float],
    current_slice: int,
    readout_history: list[list[int]],
) -> tuple[np.ndarray, float]:
    if current_slice <= 0:
        raise ValueError("current_slice must be at least 1.")
    if len(readout_history) != current_slice:
        raise ValueError(
            "readout_history must contain exactly one readout vector for each past slice."
        )

    total_qubits = n1 * m
    state = build_isqnn_pre_measurement_state(
        bitstring=bitstring,
        n1=n1,
        m=m,
        theta_list=theta_list,
        omit_local_slice=current_slice,
    )

    history_probability = 1.0
    for slice_index in range(current_slice):
        outcomes = readout_history[slice_index]
        if len(outcomes) != m:
            raise ValueError(
                f"Slice {slice_index} readout length {len(outcomes)} does not match m={m}."
            )
        for offset, outcome in enumerate(outcomes):
            qubit = slice_index * m + offset
            state, probability = measure_x_with_forced_outcome(
                state, qubit, outcome, total_qubits
            )
            history_probability *= probability

    keep_qubits = list(range(current_slice * m, (current_slice + 1) * m))
    rho_pre_local = reduced_density_matrix(state, keep_qubits, total_qubits)
    return rho_pre_local, history_probability


def compare_layer_states(
    dqnn_trace: DQNNLayerTrace,
    rho_pre_local: np.ndarray,
    local_unitary: np.ndarray,
    tol: float,
) -> dict:
    rho_post_local = local_unitary @ rho_pre_local @ local_unitary.conj().T

    pre_fidelity = pure_state_fidelity(dqnn_trace.pre_local_state, rho_pre_local)
    post_fidelity = pure_state_fidelity(dqnn_trace.post_local_state, rho_post_local)
    pre_purity = density_purity(rho_pre_local)
    post_purity = density_purity(rho_post_local)
    pre_distance = frobenius_distance_to_pure_state(
        dqnn_trace.pre_local_state, rho_pre_local
    )
    post_distance = frobenius_distance_to_pure_state(
        dqnn_trace.post_local_state, rho_post_local
    )

    return {
        "pre_same": pre_distance <= tol,
        "post_same": post_distance <= tol,
        "pre_fidelity": pre_fidelity,
        "post_fidelity": post_fidelity,
        "pre_purity": pre_purity,
        "post_purity": post_purity,
        "pre_distance": pre_distance,
        "post_distance": post_distance,
    }


def format_bits(bits: list[int]) -> str:
    return "".join(str(bit) for bit in bits)


def parse_theta_argument(theta_json: str | None, total_length: int, seed: int) -> list[float]:
    if theta_json is None:
        rng = np.random.default_rng(seed)
        return [float(value) for value in (rng.random(total_length) * np.pi)]

    parsed = json.loads(theta_json)
    return normalize_theta_list(parsed, total_length)


def validate_bitstring(bitstring: str, n1: int, m: int) -> None:
    if len(bitstring) != n1 * m:
        raise ValueError(
            f"bitstring length {len(bitstring)} must equal n1*m = {n1 * m}."
        )
    if any(bit not in {"0", "1"} for bit in bitstring):
        raise ValueError("bitstring must contain only '0' and '1'.")


def run_single_trial(
    bitstring: str,
    n1: int,
    m: int,
    theta_list: list[float],
    trajectory_seed: int,
    tol: float,
) -> dict:
    _, dqnn_traces, final_x_readout = trace_dqnn_layers(
        bitstring=bitstring,
        n1=n1,
        m=m,
        theta_list=theta_list,
        trajectory_seed=trajectory_seed,
    )

    layer_results = []
    readout_history: list[list[int]] = []
    for dqnn_trace in dqnn_traces:
        readout_history.append(dqnn_trace.readout)
        rho_pre_local, history_probability = conditional_isqnn_slice_density(
            bitstring=bitstring,
            n1=n1,
            m=m,
            theta_list=theta_list,
            current_slice=dqnn_trace.layer_index,
            readout_history=readout_history,
        )
        local_unitary = slice_local_unitary(theta_list, dqnn_trace.layer_index, m)
        comparison = compare_layer_states(
            dqnn_trace=dqnn_trace,
            rho_pre_local=rho_pre_local,
            local_unitary=local_unitary,
            tol=tol,
        )
        layer_results.append(
            {
                "layer_index": dqnn_trace.layer_index,
                "readout": dqnn_trace.readout,
                "history_probability": history_probability,
                **comparison,
            }
        )

    return {
        "final_x_readout": final_x_readout,
        "layers": layer_results,
    }


def print_trial_report(trial_index: int, report: dict) -> None:
    print(f"Trial {trial_index}")
    for layer in report["layers"]:
        layer_number = layer["layer_index"]
        readout = format_bits(layer["readout"])
        history_probability = layer["history_probability"]
        print(
            f"  Layer {layer_number}: readout={readout}, "
            f"ISQNN_history_prob={history_probability:.12g}"
        )
        print(
            f"    pre_local  same={layer['pre_same']}, "
            f"fidelity={layer['pre_fidelity']:.12g}, "
            f"purity={layer['pre_purity']:.12g}, "
            f"fro_distance={layer['pre_distance']:.12g}"
        )
        print(
            f"    post_local same={layer['post_same']}, "
            f"fidelity={layer['post_fidelity']:.12g}, "
            f"purity={layer['post_purity']:.12g}, "
            f"fro_distance={layer['post_distance']:.12g}"
        )
    print(f"  Final DQNN X readout: {format_bits(report['final_x_readout'])}")
    print()


def summarize_reports(reports: list[dict]) -> None:
    total_layers = sum(len(report["layers"]) for report in reports)
    pre_same_count = sum(
        1 for report in reports for layer in report["layers"] if layer["pre_same"]
    )
    post_same_count = sum(
        1 for report in reports for layer in report["layers"] if layer["post_same"]
    )
    print("Summary")
    print(f"  Compared layers: {total_layers}")
    print(f"  pre_local exact matches within tolerance: {pre_same_count}")
    print(f"  post_local exact matches within tolerance: {post_same_count}")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the layer-wise effective states of DQNN and ISQNN under the "
            "same measurement/readout history."
        )
    )
    parser.add_argument("--bitstring", required=True, help="Binary string of length n1*m.")
    parser.add_argument("--n1", required=True, type=int, help="Number of layers/slices.")
    parser.add_argument("--m", required=True, type=int, help="Qubits per layer/slice.")
    parser.add_argument(
        "--theta-json",
        default=None,
        help="JSON scalar or JSON list for theta values. If omitted, a random list is used.",
    )
    parser.add_argument(
        "--theta-seed",
        type=int,
        default=42,
        help="Seed used when theta-json is omitted.",
    )
    parser.add_argument(
        "--trajectory-seed",
        type=int,
        default=1234,
        help="Seed used for the sampled DQNN readout trajectory.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=1,
        help="Number of independent DQNN trajectories to compare.",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-9,
        help="Tolerance on the density-matrix Frobenius distance.",
    )
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    validate_bitstring(args.bitstring, args.n1, args.m)
    total_length = args.n1 * args.m
    theta_list = parse_theta_argument(args.theta_json, total_length, args.theta_seed)

    print("Parameters")
    print(f"  bitstring        = {args.bitstring}")
    print(f"  n1               = {args.n1}")
    print(f"  m                = {args.m}")
    print(f"  theta_length     = {len(theta_list)}")
    print(f"  theta_preview    = {[round(value, 6) for value in theta_list[:min(6, len(theta_list))]]}")
    print(f"  trajectory_seed  = {args.trajectory_seed}")
    print(f"  trials           = {args.trials}")
    print(f"  tolerance        = {args.tol}")
    print()

    reports = []
    for trial_offset in range(args.trials):
        report = run_single_trial(
            bitstring=args.bitstring,
            n1=args.n1,
            m=args.m,
            theta_list=theta_list,
            trajectory_seed=args.trajectory_seed + trial_offset,
            tol=args.tol,
        )
        reports.append(report)
        print_trial_report(trial_offset + 1, report)

    summarize_reports(reports)


if __name__ == "__main__":
    main()
