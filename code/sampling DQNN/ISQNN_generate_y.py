import cirq
import numpy as np
import random


def ISQNN_generate_y(bitstring, n1, m, theta_list):
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
            if slice_bits[i] == '0':
                full_circuit.append(cirq.H(qubits[slice_idx][i]))  # |+⟩
            # 如果是 '1'，保持 |0⟩

    # --- Step 2: 对每个 slice 应用内部操作 (Rz 和 CZ) ---
    for slice_idx in range(n1):
        slice_qubits = qubits[slice_idx]  # 该slice的所有qubits

        # Rz 旋转 (使用对应 slice 的 theta 参数)
        start_theta = slice_idx * m
        for i in range(m):
            theta_idx = start_theta + i
            full_circuit.append(cirq.rz(full_theta_list[theta_idx])(slice_qubits[i]))

        # CZ 门 (slice 内部两两连接)
        for i in range(m // 2):
            full_circuit.append(cirq.CZ(slice_qubits[2*i], slice_qubits[2*i + 1]))

    # --- Step 3: 在相邻 slice 之间应用 entanglement (CZ 门) ---
    for slice_idx in range(n1 - 1):
        for i in range(m):
            current_qubit = qubits[slice_idx][i]
            next_qubit = qubits[slice_idx + 1][i]
            full_circuit.append(cirq.CZ(current_qubit, next_qubit))

    # --- Step 4: Shadow QNN 处理 - 对所有 slice 进行测量 ---
    for slice_idx in range(n1):
        for i in range(m):
            qubit_to_measure = qubits[slice_idx][i]  # 第slice_idx个 slice 的 qubit i

            # 对该 slice 的 qubit i 进行 X 测量
            full_circuit.append(cirq.H(qubit_to_measure))  # X 测量 = H + Z 测量
            full_circuit.append(cirq.measure(qubit_to_measure, key=f'meas_slice{slice_idx}_q{i}'))

    # 运行电路并获取真实的测量结果
    simulator = cirq.Simulator()
    result = simulator.run(full_circuit, repetitions=1)

    # 从测量结果中提取y
    y = []
    for slice_idx in range(n1):
        for i in range(m):
            key = f'meas_slice{slice_idx}_q{i}'
            meas_result = result.measurements[key][0][0]  # 获取测量结果#第一个0是因为只有一次重复，第二个0是因为每个key中只对一个key做测量
            y.append(int(meas_result))  # 转换为int并添加到y中

    return full_circuit, y


if __name__ == "__main__":
    # 示例参数
    bitstring = "000100"  
    n1 = 3
    m = 2
    n = n1 * m
    theta_test = [random.uniform(0, 1) * np.pi for _ in range(n)]

    circuit, y = ISQNN_generate_y(bitstring, n1, m, theta_test)

    print("Theta parameters (first 8):", theta_test[:8])
    print("Generated y:", y)
    print("\n=== 完整的量子电路 ===")
    print(circuit)
