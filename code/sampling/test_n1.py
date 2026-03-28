import cirq
import numpy as np
import random
from ISQNN_generate_y import ISQNN_generate_y

def test_different_n1():
    """测试不同n1值的ISQNN_generate_y函数"""

    # 测试参数
    m = 4  # 每层4个qubits

    # 测试n1=2的情况
    print("=== 测试 n1=2 ===")
    bitstring_2 = "00110101"  # 长度8
    n1_2 = 2
    theta_2 = [random.uniform(0, 1) * np.pi for _ in range(n1_2 * m)]

    circuit_2, y_2 = ISQNN_generate_y(bitstring_2, n1_2, m, theta_2)
    print(f"bitstring: {bitstring_2}")
    print(f"y: {y_2}")
    print(f"电路深度: {len(circuit_2)}")
    print()

    # 测试n1=3的情况
    print("=== 测试 n1=3 ===")
    bitstring_3 = "001101010011"  # 长度12
    n1_3 = 3
    theta_3 = [random.uniform(0, 1) * np.pi for _ in range(n1_3 * m)]

    circuit_3, y_3 = ISQNN_generate_y(bitstring_3, n1_3, m, theta_3)
    print(f"bitstring: {bitstring_3}")
    print(f"y: {y_3}")
    print(f"电路深度: {len(circuit_3)}")
    print()

    # 测试n1=4的情况
    print("=== 测试 n1=4 ===")
    bitstring_4 = "0011010100110011"  # 长度16
    n1_4 = 4
    theta_4 = [random.uniform(0, 1) * np.pi for _ in range(n1_4 * m)]

    circuit_4, y_4 = ISQNN_generate_y(bitstring_4, n1_4, m, theta_4)
    print(f"bitstring: {bitstring_4}")
    print(f"y: {y_4}")
    print(f"电路深度: {len(circuit_4)}")
    print()

    print("所有测试完成！")

if __name__ == "__main__":
    test_different_n1()