from pathlib import Path
import numpy as np
import random
import time
from tqdm import tqdm
from sampling.DQNN_generate_y import DQNN_generate_y
from sampling.ISQNN_generate_y import idqnn_connectivity
from MNIST.Encode_0to9 import encode_diagonal

ROOT = Path(__file__).resolve().parents[2]
MNIST_DIR = Path(__file__).resolve().parent

n1 = 10
m = 10
threshold = 0.4
random.seed(42)

theta_path = MNIST_DIR / "data" / f"estimate_theta_binarized{threshold}.npz"
if not theta_path.exists():
	raise FileNotFoundError(f"未找到参数文件: {theta_path}")
with np.load(theta_path) as data:
	theta_hat_matrix = data["theta_hat_matrix"]
theta_hat_matrix_flat = theta_hat_matrix.reshape(-1)

print(f"theta_hat_matrix shape: {theta_hat_matrix.shape}")

X_test=np.load(MNIST_DIR / "data" / "test_MNIST_10x10_binarize0.5.npy")
Y_test=np.load(MNIST_DIR / "data" / "test_MNIST_labels.npy")
X_test = X_test.reshape(X_test.shape[0], -1)
X_test = X_test.astype(int)
X_flip = 1 - X_test# 01反转，由于不反转时0的比例太多了，1的比例太少了。
x_test_bitstrings=np.array(["".join(map(str, row)) for row in X_flip], dtype=str) #将每个样本的像素值（0 或 1）转换为字符串，并将这些字符串存储在一个新的 NumPy 数组 x_bitstrings 中。每个元素都是一个长度为 100 的字符串，表示一个样本的二值图像数据。
y_test_encoded= np.array([encode_diagonal(label) for label in Y_test])#不压平，保持10x10的形状，方便后续处理。

print(f"x_test_bitstrings shape: {x_test_bitstrings.shape}, dtype: {x_test_bitstrings.dtype}")
print(f"y_test_encoded shape: {y_test_encoded.shape}, dtype: {y_test_encoded.dtype}")

y_inferred = []

for bitsreing in tqdm(x_test_bitstrings):
	_,y=DQNN_generate_y(bitsreing, n1, m, theta_hat_matrix_flat)
	y_inferred.append(y)
	
y_inferred = np.array(y_inferred)
print(f"y_inferred shape: {y_inferred.shape}, dtype: {y_inferred.dtype}")

output_path = MNIST_DIR / "data" / f"y_inferred_binarized{threshold}.npz"
np.savez(
	output_path,
	y_inferred=y_inferred,
	y_test_encoded=y_test_encoded,
)
print(f"Saved inferred results to: {output_path}")