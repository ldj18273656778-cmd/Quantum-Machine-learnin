# MNIST 模块说明（test_MNIST.py）

本文档用于总结 `test_MNIST.py` 的核心流程，方便后续快速理解代码功能。

## 代码目标

对 10×10 二值化 MNIST 数据进行逐比特参数估计，得到 `theta` 的平铺向量和矩阵形式，并将结果保存到 `data/estimate_theta1.npz`。

## 主要流程

1. **读取数据**
   - 图像：`data/MNIST_10x10_binarize0.5.npy`
   - 标签：`data/MNIST_labels.npy`

2. **数据预处理**
   - 将图像展平为长度 100 的向量。
   - 将像素值转为整型（0/1）。
   - 执行 `X_flip = 1 - X`，把 0/1 反转。
   - 将每个样本转换为长度 100 的二进制字符串 `x_bitstrings`。
   - 将标签通过 `encode_diagonal()` 编码为长度 100 的 `y_encoded`。

3. **构建图结构与邻接表**
   - 使用 `idqnn_connectivity(n1=10, m=10)` 生成连接关系。
   - 用 `build_adjacency()` 构建每个 bit 的邻居集合。

4. **按 bit 估计参数**
   - 遍历 100 个 bit。
   - 调用 `estimate_theta_from_filtered_samples()`。
   - 若出现 `N_sp = 0`（无满足条件样本），则**跳过该 bit**并记录到 `skipped_bits`。
   - 否则写入该 bit 的估计值到 `theta_hat_flat[target_bit]`。

5. **结果组织与保存**
   - `theta_hat_flat` reshape 成 `theta_hat_matrix (10×10)`。
   - 保存为：`data/estimate_theta1.npz`。

## 输出文件说明（estimate_theta1.npz）

包含以下键：

- `theta_hat_flat`：长度 100 的估计结果。
- `theta_hat_matrix`：10×10 的估计矩阵。
- `skipped_bits`：因 `N_sp=0` 被跳过的 bit 索引（0-based）。

## 运行后的终端信息

脚本会打印：

- 数据形状与样例。
- 指定位（示例 `target_bit=74`）的邻居和满足条件样本数。
- 最终 `theta_hat_matrix`。
- 跳过 bit 的数量与索引。
- 结果保存路径。
