import numpy as np
import random


def DQNN_generate_y(bitstring, n1, m, theta_list):
    try:
        import cirq
    except Exception as e:
        cirq = None
        print(f"Warning: cirq import failed in DQNN_generate_y: {e}")

    if cirq is None:
        # fallback: generate random bits of length n1*m
        y = [random.randint(0,1) for _ in range(n1*m)]
        return None, y
    qubits = cirq.LineQubit.range(m)# 创建 m 个量子比特
    sim = cirq.Simulator()# 创建模拟器
    
    # 确保theta_list是列表格式，如果是标量则转换
    if isinstance(theta_list, (int, float)):
        # 如果theta_list是标量，为每个block创建参数列表
        full_theta_list = [theta_list] * (n1 * m)
    else:
        full_theta_list = theta_list

    # --- Step 1: 初始态 φ ---
    full_circuit = cirq.Circuit()

    # 编码前 m bits
    for i in range(m):
        if bitstring[i] == '0':
            full_circuit.append(cirq.H(qubits[i]))

    # Rz
    for i in range(m):
        full_circuit.append(cirq.rz(full_theta_list[i])(qubits[i]))

    # CZ
    for i in range((m )//2):# 只连接偶数索引的qubits，形成CZ门
        full_circuit.append(cirq.CZ(qubits[2*i], qubits[2*i + 1]))

    # 得到 φ^(1)
    result = sim.simulate(full_circuit)# 模拟电路，得到最终状态
    state = result.final_state_vector

    y = []

    # --- Step 2: 逐块处理 ---
    for block in range(1, n1):

        for i in range(m):
            theta_index = block * m + i
            x_bit = bitstring[block * m + i]

            if x_bit == '0':
                # 先应用H门
                h_circuit = cirq.Circuit()# 创建一个新的电路
                h_circuit.append(cirq.H(qubits[i]))# 无论y的值是什么，都添加H门到电路
                full_circuit.append(cirq.H(qubits[i]))# 记录到完整电路
                
                result = sim.simulate(h_circuit, initial_state=state, qubit_order=qubits)
                state = result.final_state_vector# 更新状态
                
                # 生成均匀分布的随机值 (0 或 1)
                meas = random.randint(0, 1)
                y.append(meas)

                # 如果随机值为1，加Z门
                if meas == 1:
                    z_circuit = cirq.Circuit(cirq.Z(qubits[i]))
                    full_circuit.append(cirq.Z(qubits[i]))
                    result = sim.simulate(z_circuit, initial_state=state, qubit_order=qubits)
                    state = result.final_state_vector

            else:  # x_bit == '1'
                # 测量 X：先应用H门，然后测量
                h_circuit = cirq.Circuit()
                h_circuit.append(cirq.measure(cirq.X(qubits[i]), key='meas'))# 测量X并记录结果至meas
                result = sim.simulate(h_circuit, initial_state=state, qubit_order=qubits)
                meas = int(result.measurements['meas'][0])

                y.append(meas)
                
                # 记录到完整电路
                full_circuit.append(cirq.H(qubits[i]))
                full_circuit.append(cirq.measure(qubits[i], key=f'meas_b{block}_q{i}'))

                # 重置为 |0>
                reset_circuit = cirq.Circuit(
                    cirq.ResetChannel().on(qubits[i])
                )
                result = sim.simulate(reset_circuit, initial_state=state, qubit_order=qubits)
                state = result.final_state_vector
                full_circuit.append(cirq.ResetChannel().on(qubits[i]))

        # --- 每个block处理完后，应用 Rz 和 CZ （除开最后一步前）---
        if block < n1 - 1:
            block_circuit = cirq.Circuit()
            
            # Rz 旋转 - 使用该block对应的theta参数
            for i in range(m):
                theta_index = block * m + i
                block_circuit.append(cirq.rz(full_theta_list[theta_index])(qubits[i]))
                full_circuit.append(cirq.rz(full_theta_list[theta_index])(qubits[i]))
            
            # CZ 门
            #for i in range(m - 1):
            #    block_circuit.append(cirq.CZ(qubits[i], qubits[i + 1]))
            #    full_circuit.append(cirq.CZ(qubits[i], qubits[i + 1]))
            for i in range((m )//2):
                block_circuit.append(cirq.CZ(qubits[2*i], qubits[2*i + 1]))
                full_circuit.append(cirq.CZ(qubits[2*i], qubits[2*i + 1]))



            result = sim.simulate(block_circuit, initial_state=state, qubit_order=qubits)
            state = result.final_state_vector

    # --- Step 3: 最后一轮：全部测量 X ---
    final_measure_circuit = cirq.Circuit()

    for i in range(m):
        final_measure_circuit.append(cirq.H(qubits[i]))
        final_measure_circuit.append(cirq.measure(qubits[i], key=f'f{i}'))
        full_circuit.append(cirq.H(qubits[i]))
        full_circuit.append(cirq.measure(qubits[i], key=f'f{i}'))

    result = sim.simulate(final_measure_circuit, initial_state=state, qubit_order=qubits)

    for i in range(m):
        y.append(int(result.measurements[f'f{i}'][0]))

    return full_circuit, y

if __name__ == "__main__":
    # 示例参数
    bitstring = "00001111000000110000"  # n = n1*m
    n1 = 4
    m = 5
    n = n1 * m
    theta_test = [random.uniform(0, 1)*np.pi for _ in range(n)]# 随机生成theta参数列表

    circuit, y = DQNN_generate_y(bitstring, n1, m, theta_test)
    
    print("Theta parameters:", theta_test)
    print("Generated y:", y)
    print("\n=== 完整的量子电路 ===")
    print(circuit)