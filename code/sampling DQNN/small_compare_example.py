from compare_effective_states import format_bits, run_single_trial


def main() -> None:
    # 小规模案例: n1=3 层, 每层 m=2 个 qubit
    bitstring = "000100"
    n1 = 3
    m = 2
    theta_list = [0.2, 0.6, 1.0, 1.4, 1.8, 2.2]
    trajectory_seed = 11
    tol = 1e-9

    report = run_single_trial(
        bitstring=bitstring,
        n1=n1,
        m=m,
        theta_list=theta_list,
        trajectory_seed=trajectory_seed,
        tol=tol,
    )

    print("Small-scale example")
    print(f"bitstring = {bitstring}")
    print(f"n1 = {n1}, m = {m}")
    print(f"theta_list = {theta_list}")
    print(f"trajectory_seed = {trajectory_seed}")
    print()

    for layer in report["layers"]:
        print(f"Layer {layer['layer_index']}")
        print(f"  readout        = {format_bits(layer['readout'])}")
        print(f"  history_prob   = {layer['history_probability']:.12g}")
        print(f"  pre_same       = {layer['pre_same']}")
        print(f"  pre_fidelity   = {layer['pre_fidelity']:.12g}")
        print(f"  pre_purity     = {layer['pre_purity']:.12g}")
        print(f"  pre_distance   = {layer['pre_distance']:.12g}")
        print(f"  post_same      = {layer['post_same']}")
        print(f"  post_fidelity  = {layer['post_fidelity']:.12g}")
        print(f"  post_purity    = {layer['post_purity']:.12g}")
        print(f"  post_distance  = {layer['post_distance']:.12g}")
        print()

    print(f"Final DQNN X readout = {format_bits(report['final_x_readout'])}")


if __name__ == "__main__":
    main()
