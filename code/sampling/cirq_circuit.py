import cirq
import numpy as np


def create_initial_circuit(bitstring):
    """
    根据 bitstring 创建初始量子比特和电路。
    bitstring 中的 '0' 对应 Hadamard 门，初始化为叠加态；其他位保持 |0⟩。
    """
    m = len(bitstring)
    qubits = cirq.LineQubit.range(m)
    circuit = cirq.Circuit()
    
    # 编码：对 '0' 应用 H 门
    for i in range(m):
        if bitstring[i] == '0':
            circuit.append(cirq.H(qubits[i]))
    
    return qubits, circuit


def add_quantum_operations(circuit, qubits, theta, layers=1):
    """
    向电路添加量子操作：Rz 旋转和 CZ 门。
    可以指定层数以实现多层操作。
    """
    for layer in range(layers):
        # Rz 旋转
        if isinstance(theta, (int, float)):
            theta_list = [theta] * len(qubits)
        else:
            theta_list = theta
        
        for i, qubit in enumerate(qubits):
            circuit.append(cirq.rz(theta_list[i])(qubit))
        
        # CZ 门
        for i in range(len(qubits) - 1):
            circuit.append(cirq.CZ(qubits[i], qubits[i + 1]))
    
    return circuit


def add_measurement(circuit, qubits):
    """
    向电路添加测量操作。
    """
    circuit.append(cirq.measure(*qubits, key='result'))
    return circuit


def build_circuit(bitstring, theta, layers=1, measure=True):
    """
    构建完整电路：初始化 + 操作 + 可选测量。
    """
    qubits, circuit = create_initial_circuit(bitstring)
    circuit = add_quantum_operations(circuit, qubits, theta, layers)
    if measure:
        circuit = add_measurement(circuit, qubits)
    return circuit


# ⭐ 关键：主函数入口（用于直接运行脚本）
if __name__ == "__main__":
    bitstring = "010110"
    theta = np.pi / 4
    layers = 1  # 可以调整层数
    
    circuit = build_circuit(bitstring, theta, layers, measure=True)
    print(circuit)