import numpy as np
import random
import matplotlib.pyplot as plt

try:
    from tqdm import trange
except Exception:
    trange = range

# 设置 matplotlib 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

try:
    from DQNN_generate_y import DQNN_generate_y
    from ISQNN_generate_y import ISQNN_generate_y
except Exception as e:
    DQNN_generate_y = None
    ISQNN_generate_y = None
    print(f"Warning: failed to import DQNN/ISQNN generator: {e}")


def bitlist_to_decimal(y_bits):
    # 假设 y_bits 是高位在前的二进制表示
    return int(''.join(str(int(b)) for b in y_bits), 2)


def generate_probability_distribution(bitstring, n1, m, theta_list, num_samples=100, bin_width=1):
    dqnn_samples = []
    isqnn_samples = []

    if not callable(DQNN_generate_y) or not callable(ISQNN_generate_y):
        raise ImportError(
            "DQNN_generate_y or ISQNN_generate_y is unavailable. Please check the imports and Cirq installation."
        )

    for _ in trange(num_samples, desc="Sampling", leave=True):#这一段代码使用了 tqdm 库来显示采样进度条，增加了用户体验。每次迭代都会调用 DQNN_generate_y 和 ISQNN_generate_y 函数来生成对应的 y 值，并将其转换为十进制后存储在列表中。最后，使用 numpy 的 histogram 函数计算直方图数据，并返回样本列表和直方图数据。
        _, y_dqnn_bits = DQNN_generate_y(bitstring, n1, m, theta_list)
        _, y_isqnn_bits = ISQNN_generate_y(bitstring, n1, m, theta_list)

        y_dqnn = bitlist_to_decimal(y_dqnn_bits)
        y_isqnn = bitlist_to_decimal(y_isqnn_bits)

        dqnn_samples.append(y_dqnn)
        isqnn_samples.append(y_isqnn)

    n = n1 * m
    max_val = 2**n
    bins = np.arange(0, max_val + bin_width, bin_width)
    dqnn_hist, _ = np.histogram(dqnn_samples, bins=bins, range=(0, max_val))
    isqnn_hist, _ = np.histogram(isqnn_samples, bins=bins, range=(0, max_val))
    return dqnn_samples, isqnn_samples, dqnn_hist, isqnn_hist, bins


if __name__ == "__main__":
    bitstring = "000000000000"
    n1 = 3
    m = 4
    n = n1 * m
    theta_test = [random.uniform(0, 1) * np.pi for _ in range(n)]

    dqnn_samples, isqnn_samples, dqnn_hist, isqnn_hist, bins = generate_probability_distribution(
        bitstring, n1, m, theta_test, num_samples=10000, bin_width=50
    )

    print("DQNN sample decimal y:", dqnn_samples[:20], "...")
    print("ISQNN sample decimal y:", isqnn_samples[:20], "...")
    print("DQNN hist:", dqnn_hist)
    print("ISQNN hist:", isqnn_hist)

    plt.figure(figsize=(8, 4))
    plt.hist(dqnn_samples, bins=bins, alpha=0.6, label="DQNN")
    plt.hist(isqnn_samples, bins=bins, alpha=0.6, label="ISQNN")
    plt.xlabel("y (decimal)")
    plt.ylabel("count")
    plt.title(f"DQNN/ISQNN Histogram (Decimal) for bitstring: {bitstring}")
    plt.legend()
    plt.grid(True)
    # plt.savefig('dqnn_isqnn_histogram.png')
    plt.show()
