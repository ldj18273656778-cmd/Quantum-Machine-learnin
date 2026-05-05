from pathlib import Path
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
MNIST_DIR = ROOT / "code" / "MNIST"
RX_DIR = Path(__file__).resolve().parent
CODE_DIR = ROOT / "code"

print(f"ROOT: {ROOT}")
print(f"MNIST_DIR: {MNIST_DIR}")
print(f"RX_DIR: {RX_DIR}")
print(f"CODE_DIR: {CODE_DIR}")

from sampling.ISQNN_generate_y import idqnn_connectivity
from Train.find_x_indices_by_graph_condition import build_adjacency, find_indices
from MNIST.Encode_0to9 import encode_diagonal
from Train.estimate_theta_from_filtered_samples import estimate_theta_from_filtered_samples

threshold = 0.4

X=np.load(MNIST_DIR / "data" / f"MNIST_10x10_binarize{threshold}.npy")
X = X.reshape(X.shape[0], -1)#将每个样本的图像数据展平为一维数组，新的形状为 (N, 100)，其中 N 是样本数量，100 是每个样本的像素数量（10x10=100）。
X = X.astype(int)#将像素值转换为整数类型（0 或 1），以便后续处理。
print(f"X shape: {X.shape}, dtype: {X.dtype}")

y=np.load(MNIST_DIR / "data" / "MNIST_labels.npy")
print(f"y shape: {y.shape}, dtype: {y.dtype}")


print(f"Sample X[0]: {X[0]}")
print(f"Sample y[0]: {y[0]}")

X_flip = 1 - X# 01反转，由于不反转时0的比例太多了，1的比例太少了。

x_bitstrings=np.array(["".join(map(str, row)) for row in X_flip], dtype=str) #将每个样本的像素值（0 或 1）转换为字符串，并将这些字符串存储在一个新的 NumPy 数组 x_bitstrings 中。每个元素都是一个长度为 100 的字符串，表示一个样本的二值图像数据。

y_encoded= np.array([encode_diagonal(label) for label in y]).reshape(len(y), -1) #将每个标签（0-9）编码为一个 10x10 的对角线编码矩阵，并将这些矩阵展平为一维数组，新的形状为 (N, 100)，其中 N 是样本数量，100 是每个样本的编码长度（10x10=100）。

n = len(x_bitstrings[0])
n1=10
m=10


print(f"x_bitstrings shape: {x_bitstrings.shape}, dtype: {x_bitstrings.dtype}")
print(f"Sample x_bitstrings: {x_bitstrings[0:3]}")

G = idqnn_connectivity(n1, m)

target_bit = 99
adjacency = build_adjacency(n=n, edges=G["all_edges"])#生成每个比特的连通邻居列表
neighbors = sorted(adjacency[target_bit])

indices = find_indices(x=x_bitstrings, target_bit=target_bit, adjacency=adjacency)

print(f"Neighbors (0-based): {neighbors}")
print(f"Number of samples satisfying target_bit condition: {len(indices)}")
print(f"3 Indices of x_bitstrings satisfying target_bit condition: {indices[:3]}")
print(f"Adjacency list for target_bit {target_bit}: {adjacency[target_bit]}")

# Number_of_satisfying=[]
# for target_bit in range(n):
#     indices = find_indices(x=x_bitstrings, target_bit=target_bit, adjacency=adjacency)
#     Number_of_satisfying.append(len(indices))

# import matplotlib.pyplot as plt

# y = Number_of_satisfying

# plt.figure(figsize=(6,4))
# plt.plot(np.arange(1, len(y) + 1), y)

# plt.xlabel("Index (1 to 100)")
# plt.ylabel("Number_of_satisfying")
# plt.title("Visualization of Number_of_satisfying")

# plt.grid(True)
# plt.show()

theta_hat_flat = np.zeros(n, dtype=float)
records: list[dict] = []
skipped_bits: list[int] = []

bit_iter = range(n)
if tqdm is not None:
    bit_iter = tqdm(bit_iter, total=n, desc="Estimating all thetas", unit="bit")

for target_bit in bit_iter:
    try:
        result = estimate_theta_from_filtered_samples(
            x=x_bitstrings,
            y=y_encoded,
            target_bit=target_bit,
            adjacency=adjacency,
            show_progress=False,
        )
    except ValueError as e:
        if "N_sp = 0" in str(e):
            skipped_bits.append(target_bit)
            continue
        raise

    theta_hat_flat[target_bit] = result["theta_hat_rad"]
    records.append(result)

theta_hat_matrix = theta_hat_flat.reshape(n1, m)

output_path = RX_DIR / "data" / f"estimate_theta_binarized{threshold}.npz"
np.savez(
    output_path,
    theta_hat_flat=theta_hat_flat,
    theta_hat_matrix=theta_hat_matrix,
    skipped_bits=np.asarray(skipped_bits, dtype=int),
)

print("theta_hat matrix (rad):")
print(theta_hat_matrix)
print(f"Skipped bits (N_sp=0): {len(skipped_bits)}")
if skipped_bits:
    print(f"Skipped bit indices (0-based): {skipped_bits}")
print(f"Saved results to: {output_path}")
