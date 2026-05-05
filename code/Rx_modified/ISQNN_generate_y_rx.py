import numpy as np
import random


def idqnn_connectivity(n1, m):
    """返回网络连通结构 G（所有 CZ 门作用的 bit 对）。

    bit 编号采用展平索引: idx = slice_idx * m + i。
    """

    def idx(slice_idx, i):
        return slice_idx * m + i

    intra_slice_edges = []
    for slice_idx in range(n1):
        if slice_idx % 2 == 0:
            for i in range(m // 2):
                intra_slice_edges.append(
                    (idx(slice_idx, 2 * i), idx(slice_idx, 2 * i + 1))
                )
        else:
            for i in range((m - 1) // 2):
                intra_slice_edges.append(
                    (idx(slice_idx, 2 * i + 1), idx(slice_idx, 2 * i + 2))
                )

    inter_slice_edges = []
    for slice_idx in range(n1 - 1):
        for i in range(m):
            inter_slice_edges.append((idx(slice_idx, i), idx(slice_idx + 1, i)))

    G = {
        "n1": n1,
        "m": m,
        "n": n1 * m,
        "intra_slice_edges": intra_slice_edges,
        "inter_slice_edges": inter_slice_edges,
        "all_edges": intra_slice_edges + inter_slice_edges,
    }  # G 是一个字典，包含了网络的层数 n1、每层的 qubit 数 m、总 qubit 数 n，以及所有 CZ 门作用的 bit 对（intra_slice_edges 和 inter_slice_edges）。
    # G[""all_edges"] 是一个列表，包含了所有 CZ 门作用的 bit 对，先是每层内部的连接（intra_slice_edges），然后是相邻层之间的连接（inter_slice_edges）。
    return G


def ISQNN_generate_y_rx(bitstring, n1, m, theta_list, rx_angle=0.0):
    try:
        import cirq
    except Exception as e:
        cirq = None
        print(f"Warning: cirq import failed in ISQNN_generate_y_rx: {e}")

    if cirq is None:
        y = [random.randint(0, 1) for _ in range(n1 * m)]
        return None, y
    """
    Shadow QNN (ISQNN) 生成 bitstring y 的函数 - 适用于任意 n1 层

    参数:
    - bitstring: 长度为 n = n1*m 的二进制字符串
    - n1: 层数 (slice 数)
    - m: 每层的 qubit 数
    - theta_list: 旋转参数列表，长度应为 n1*m

    返回:
    - full_circuit: 完整的量子电路
    - y: 生成的 bitstring
    """
    n = n1 * m

    # 验证输入参数
    if len(bitstring) != n:
        raise ValueError(f"bitstring长度({len(bitstring)})必须等于n1*m({n})")

    # 创建矩形 qubit 结构
    # 将 n1*m 个 qubits 排列成矩形 (rows x cols)
    # rows = n1 (层数), cols = m (每层的 qubit 数)
    rows = n1
    cols = m

    # 使用二维列表存储qubits，便于按层(slice)和位置索引
    qubits = [[cirq.GridQubit(r, c) for c in range(cols)] for r in range(rows)]

    # 确保 theta_list 是列表格式且长度正确
    if isinstance(theta_list, (int, float)):
        full_theta_list = [theta_list] * n
    else:
        full_theta_list = theta_list
        if len(full_theta_list) < n:
            # 如果 theta_list 太短，用最后一个值填充
            full_theta_list.extend([full_theta_list[-1]] * (n - len(full_theta_list)))
        elif len(full_theta_list) > n:
            # 如果 theta_list 太长，截断
            full_theta_list = full_theta_list[:n]

    full_circuit = cirq.Circuit()
    y = []

    # --- Step 1: 初始化所有 n1 个 slice ---
    for slice_idx in range(n1):
        start_bit = slice_idx * m
        end_bit = (slice_idx + 1) * m
        slice_bits = bitstring[start_bit:end_bit]

        for i in range(m):
            if slice_bits[i] == "0":
                full_circuit.append(cirq.H(qubits[slice_idx][i]))  # |+⟩
            # 如果是 '1'，保持 |0⟩

    # --- Step 1.5: Rx rotation (mix information globally) ---
    if rx_angle != 0.0:
        for slice_idx in range(n1):
            for i in range(m):
                full_circuit.append(cirq.rx(rx_angle)(qubits[slice_idx][i]))

    # --- Step 2: 对每个 slice 应用内部操作 (Rz 和 CZ) ---
    for slice_idx in range(n1):
        slice_qubits = qubits[slice_idx]  # 该slice的所有qubits

        # Rz 旋转 (使用对应 slice 的 theta 参数)
        start_theta = slice_idx * m
        for i in range(m):
            theta_idx = start_theta + i
            full_circuit.append(cirq.rz(full_theta_list[theta_idx])(slice_qubits[i]))

        # CZ 门 (slice 内部按奇偶 slice 交替连接，与 DQNN 保持一致)
        if slice_idx % 2 == 0:  # 偶数 slice: (0,1), (2,3), ...
            for i in range(m // 2):
                full_circuit.append(
                    cirq.CZ(slice_qubits[2 * i], slice_qubits[2 * i + 1])
                )
        else:  # 奇数 slice: (1,2), (3,4), ...
            for i in range((m - 1) // 2):
                full_circuit.append(
                    cirq.CZ(slice_qubits[2 * i + 1], slice_qubits[2 * i + 2])
                )

    # --- Step 3: 在相邻 slice 之间应用 entanglement (CZ 门) ---
    for slice_idx in range(n1 - 1):
        for i in range(m):
            current_qubit = qubits[slice_idx][i]
            next_qubit = qubits[slice_idx + 1][i]
            full_circuit.append(cirq.CZ(current_qubit, next_qubit))

    # --- 执行 Step 1-3，获得初始状态向量 ---
    simulator = cirq.Simulator()
    result = simulator.simulate(full_circuit)
    state = result.final_state_vector

    # 将所有 qubits 展平成一维列表，用于 qubit_order 参数
    all_qubits = [qubits[r][c] for r in range(n1) for c in range(m)]

    # --- Step 4: Shadow QNN 处理 - 对所有 slice 进行动态测量并坍缺状态 ---
    y = []
    for slice_idx in range(n1):
        for i in range(m):
            qubit_to_measure = qubits[slice_idx][i]

            # 为该 qubit 构建独立的测量电路 (H + measure)
            meas_circuit = cirq.Circuit()
            meas_circuit.append(cirq.H(qubit_to_measure))  # X 测量 = H + Z 测量
            meas_circuit.append(
                cirq.measure(qubit_to_measure, key=f"meas_slice{slice_idx}_q{i}")
            )

            # 执行测量，传入当前的量子态，获得坍缺后的新状态
            result = simulator.simulate(
                meas_circuit, initial_state=state, qubit_order=all_qubits
            )

            # 提取测量结果 (simulate 返回的结果中 measurements 是数组，提取第一个元素)
            meas_result = result.measurements[f"meas_slice{slice_idx}_q{i}"].item()
            y.append(int(meas_result))

            # 更新状态为坍缺后的状态，用于下一个 qubit 的测量
            state = result.final_state_vector

            # 将该步的测量电路添加到完整电路中（用于记录）
            full_circuit.append(meas_circuit)

    return full_circuit, y


if __name__ == "__main__":
    # 示例参数
    import time

    t0 = time.perf_counter()

    bitstring = "1000100010000111"
    n1 = 4
    m = 4
    n = n1 * m
    theta_test = [random.uniform(0, 1) * np.pi for _ in range(n)]

    circuit, y = ISQNN_generate_y_rx(bitstring, n1, m, theta_test)
    G = idqnn_connectivity(n1, m)

    print("Theta parameters (first 8):", theta_test[:8])
    print("Generated y:", y)
    print("CZ connectivity G:", G)
    print("\n=== 完整的量子电路 ===")
    # print(circuit)
    print(G["all_edges"])
    # print(G['all_edges'][3:5])
    t1 = time.perf_counter()
    print(f"总耗时: {t1 - t0:.6f} 秒")
