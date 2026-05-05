# Learning Parity with Noise 矩阵版项目

这个目录用于实现矩阵版 Learning Parity with Noise (LPN)，并先使用原始 ISQNN 作为后续生成模型来学习从输入 `x` 到输出 `y` 的关系。

## 问题设定

标准 LPN 中，一个输入 bitstring 为：

```text
x in {0,1}^{n_x}
```

一个 secret bitstring 为：

```text
s in {0,1}^{n_x}
```

一个输出 bit 为：

```text
y = <x, s> mod 2
```

如果需要生成 `n_y` 个输出 bit，就使用 `n_y` 个 secret vectors，把它们作为列组成 secret matrix：

```text
S in {0,1}^{n_x x n_y}
y = x S mod 2
```

对有限数据集，把很多输入按行堆叠为：

```text
X in {0,1}^{num_samples x n_x}
Y_clean = X S mod 2
Y_noisy = Y_clean XOR E
E_ij ~ Bernoulli(noise_rate)
```

训练阶段使用：

```text
(X_train, Y_train_noisy)
```

测试阶段使用新的输入：

```text
X_test
```

目标是让后续的 ISQNN 生成：

```text
Y_pred ≈ Y_test_clean
```

也就是希望模型学到 hidden secret matrix `S` 诱导的 clean parity map，而不是只拟合噪声。

## 当前直接映射约定

第一版为了直接接入 ISQNN，采用：

```text
n_x = n_y = n1 * m
```

这样每一行 `x` 可以直接作为 ISQNN 输入 bitstring，每一行 `y` 可以直接作为 ISQNN 输出 bitstring。

当前默认参数在 `config.py` 中：

```text
n_x = 16
n_y = 16
n1 = 4
m = 4
num_train = 60000
num_test = 10000
noise_rate = 0.0
seed = 47
rx_angle = pi / 4
```

建议先用 `noise_rate = 0.0` 做 sanity check。如果无噪声 parity map 都学不好，再加入 LPN 噪声就没有意义。

## 文件说明

`config.py`

统一管理实验参数和路径。

主要变量：

```text
n_x: 输入 bit 数
n_y: 输出 bit 数
n1, m: ISQNN 结构参数，要求 n_x = n_y = n1 * m
n_bits: n1 * m
num_train: 训练样本数
num_test: 测试样本数
noise_rate: 每个输出 bit 被翻转的概率
seed: 随机种子
rx_angle: Rx-modified ISQNN 使用的 Rx 角度
DATASET_PATH: LPN 数据集保存路径
THETA_PATH: 后续 theta 保存路径，当前为 data/theta_lpn.npz
THETA_RX_PATH: Rx theta 保存路径，当前为 data/theta_lpn_rx.npz
PREDICTION_PATH: 后续预测结果保存路径
PREDICTION_RX_PATH: Rx 推断结果保存路径
```

主要函数：

```text
ensure_directories()
```

创建 `data/` 和 `output_images/` 输出目录。

```text
validate_config()
```

检查配置是否合法，包括：

```text
n_x = n_y = n1 * m
0 <= noise_rate < 0.5
num_train > 0
num_test > 0
```

`gf2_utils.py`

GF(2) 基础工具函数。

```text
get_rng(seed)
```

创建 NumPy 随机数生成器。相同 `seed` 会产生相同数据，保证实验可复现。

```text
as_binary_array(values, name="array")
```

把输入转换为 `uint8` 数组，并检查所有元素只能是 `0` 或 `1`。

```text
sample_secret(n_x, n_y, rng)
```

随机生成 secret matrix：

```text
S in {0,1}^{n_x x n_y}
```

```text
sample_inputs(num_samples, n_x, rng)
```

随机生成输入矩阵：

```text
X in {0,1}^{num_samples x n_x}
```

```text
gf2_matmul(x, s)
```

计算 GF(2) 矩阵乘法：

```text
x @ s mod 2
```

对 LPN 来说就是：

```text
Y = X S mod 2
```

```text
generate_labels(x, s)
```

生成 clean labels：

```text
Y_clean = X S mod 2
```

```text
add_bernoulli_noise(labels, noise_rate, rng)
```

对标签逐 bit 加噪声：

```text
Y_noisy = Y_clean XOR noise
```

返回：

```text
Y_noisy, noise
```

```text
generate_lpn_split(num_samples, s, noise_rate, rng)
```

用固定 secret matrix `S` 生成一个 split，返回：

```text
{
  "x": X,
  "y_clean": Y_clean,
  "y_noisy": Y_noisy,
  "noise": noise,
}
```

```text
bit_accuracy(predicted, target)
```

计算 bit-level accuracy，也就是所有 bit 中预测正确的比例。

```text
sample_accuracy(predicted, target)
```

计算 sample-level accuracy，也就是整行输出 bitstring 全部正确的样本比例。

```text
hamming_distance(predicted, target)
```

计算每个样本输出 bitstring 与目标 bitstring 的 Hamming distance。

`test__LPN.py`

小型 sanity test。它会生成一个小的 `S`、`X`、`Y_clean`、`Y_noisy`，打印前几个样本，并保存：

```text
data/test_lpn_demo.npz
```

`generate_lpn_dataset.py`

正式的数据集生成脚本。

主要函数：

```text
generate_dataset(n_x, n_y, num_train, num_test, noise_rate, seed)
```

生成完整 train/test 数据集。训练集和测试集共享同一个 hidden secret matrix `S`。

返回字段：

```text
S
X_train
Y_train_clean
Y_train_noisy
train_noise
X_test
Y_test_clean
Y_test_noisy
test_noise
n_x
n_y
num_train
num_test
noise_rate
seed
```

```text
save_dataset(dataset, output_path)
```

把 dataset 保存为 `.npz` 文件。

```text
print_dataset_summary(dataset, output_path)
```

打印数据集形状、保存路径、实际翻转 bit 数等信息。

```text
main()
```

读取 `config.py` 中的参数，生成完整数据集并保存到：

```text
data/lpn_dataset.npz
```

保存文件额外包含：

```text
n1
m
n_bits
```

方便后续估计 theta 和调用 ISQNN。

`estimate_theta_lpn.py`

从 `data/lpn_dataset.npz` 中读取训练集，并引用原始 ISQNN 的学习方法估计参数 `theta`。

它使用训练数据：

```text
X_train
Y_train_noisy
```

对每个 target bit `j`，先筛选满足 ISQNN 局部图条件的样本：

```text
x_j = 0
x_neighbors = 1
```

然后调用原始估计函数：

```text
Train.estimate_theta_from_filtered_samples.estimate_theta_from_filtered_samples(...)
```

使用原始 ISQNN 学习公式估计：

```text
theta_j
```

主要函数：

```text
rows_to_bitstrings(x)
```

把二进制矩阵 `X_train` 转成 ISQNN 学习函数需要的 bitstring 数组。例如：

```text
[[1, 0, 1], [0, 1, 1]] -> ["101", "011"]
```

```text
estimate_theta_from_training_data(X_train, Y_train, n1, m)
```

对所有输出 bit 逐一估计 `theta_j`，返回：

```text
theta_hat_flat_rad
theta_hat_matrix_rad
theta_hat_flat_deg
theta_hat_matrix_deg
records
failed_bits_0based
n1
m
n_bits
model
```

其中 `records` 保存每个 bit 的筛选样本数 `N_sp`、邻居、估计角度和使用的样本索引。

```text
save_theta_estimates(estimates, output_path)
```

保存估计结果。

```text
print_estimation_summary(estimates, output_path)
```

打印估计结果摘要。

```text
main()
```

读取 `config.py` 中的路径，直接从下面文件读入数据：

```text
data/lpn_dataset.npz
```

并保存估计结果到：

```text
data/theta_lpn.npz
```

`estimate_theta_lpn_rx.py`

从 `data/lpn_dataset.npz` 中读取训练集，并引用 `Rx_modified` 中的 Rx-aware 学习方法估计参数 `theta`。

它使用训练数据：

```text
X_train
Y_train_noisy
```

筛选条件仍然是：

```text
x_j = 0
x_neighbors = 1
```

然后调用 Rx 估计函数：

```text
Rx_modified.estimate_theta_rx.estimate_theta_rx(...)
```

保存估计结果到：

```text
data/theta_lpn_rx.npz
```

`count_theta_estimation_samples.py`

只统计每个参数 `theta_j` 估计时有多少训练样本进入筛选集合，不做参数估计。

筛选条件仍然是：

```text
x_j = 0
x_neighbors = 1
```

主要函数：

```text
count_samples_for_each_theta(X_train, n1, m)
```

返回每个 target bit 的统计记录，包括：

```text
target_bit_0based
target_bit_1based
slice_idx
position_in_slice
neighbors_0based
n_neighbors
N_sp
fraction
indices_0based
```

```text
save_counts(records, output_path)
```

保存完整统计结果到 `.npz`，同时保存可读表格到 `.txt`。

```text
print_counts_table(records, output_path)
```

在终端打印每个参数对应的 `N_sp`。

```text
plot_counts(records)
```

用柱状图显示每个 `theta_j` 对应的 `N_sp`，只显示，不保存图片。

默认输出：

```text
data/theta_estimation_sample_counts.npz
data/theta_estimation_sample_counts.txt
```

`generate_test_y_lpn.py`

使用已经估计好的 `theta_lpn.npz`，调用原始 ISQNN 对测试输入 `X_test` 逐条生成输出 `Y_pred`。

主要函数：

```text
generate_predictions(X_test, theta, n1, m, show_progress=True)
```

对每一行测试输入 `x` 调用：

```text
sampling.ISQNN_generate_y.ISQNN_generate_y(...)
```

并用进度条显示推理进度。

```text
save_predictions(dataset, theta_data, Y_pred)
```

保存推理结果和测试标签，方便后续评估。

默认输出：

```text
data/lpn_test_predictions.npz
```

`generate_test_y_lpn_rx.py`

使用已经估计好的 `theta_lpn_rx.npz`，调用 Rx-modified ISQNN 对测试输入 `X_test` 逐条生成输出 `Y_pred`。

主要函数：

```text
generate_predictions_rx(X_test, theta, n1, m, rx_angle, show_progress=True)
```

对每一行测试输入 `x` 调用：

```text
Rx_modified.ISQNN_generate_y_rx.ISQNN_generate_y_rx(...)
```

并用进度条显示推理进度。

默认输出：

```text
data/lpn_test_predictions_rx.npz
```

`evaluate_lpn_generation.py`

比较 ISQNN 对测试集生成的 `Y_pred` 和底层 secret matrix `S` 生成的 clean label：

```text
Y_from_S = X_test S mod 2
```

主要函数：

```text
evaluate_predictions(Y_pred, Y_target)
```

输出指标：

```text
bit_accuracy
sample_accuracy
mean_hamming_distance
median_hamming_distance
max_hamming_distance
per_output_bit_accuracy
hamming_distance_per_sample
```

```text
save_metrics(metrics, output_path)
```

保存评估结果到 `.npz`，同时保存可读 `.txt` 报告。

默认输出：

```text
data/lpn_generation_metrics.npz
data/lpn_generation_metrics.txt
```

`evaluate_lpn_generation_rx.py`

比较 Rx-modified ISQNN 生成的 `Y_pred` 和底层 secret matrix `S` 生成的 clean label：

```text
Y_from_S = X_test S mod 2
```

默认输出：

```text
data/lpn_generation_metrics_rx.npz
data/lpn_generation_metrics_rx.txt
```

`majority_vote_single_sample.py`

对一个指定测试样本做 majority vote 实验。脚本会对同一个 `X_test[target_sample_idx]` 重复运行 ISQNN `num_shots` 次，然后对每个输出 bit 分别做多数投票。

脚本顶部有手动参数区：

```text
target_sample_idx = 0
num_shots = 11
```

`num_shots` 建议使用奇数，避免投票平局。

输出内容包括：

```text
x_test
y_true = x_test @ S mod 2
vote bit accuracy
vote hamming distance
mean/best/worst single-shot bit accuracy
每个输出 bit 的 ones/num_shots
```

## 如何运行

推荐从 `code/` 目录运行，因为当前脚本使用 package import：

```bash
cd "D:\研究生\研究生\Quantum Machine learning\code"
python -m Learning_Parity_with_Noise.test__LPN
```

生成正式训练/测试数据集：

```bash
cd "D:\研究生\研究生\Quantum Machine learning\code"
python -m Learning_Parity_with_Noise.generate_lpn_dataset
```

成功后会生成：

```text
Learning_Parity_with_Noise/data/lpn_dataset.npz
```

对训练集估计原始 ISQNN 参数：

```bash
cd "D:\研究生\研究生\Quantum Machine learning\code"
python -m Learning_Parity_with_Noise.estimate_theta_lpn
```

成功后会生成：

```text
Learning_Parity_with_Noise/data/theta_lpn.npz
```

对训练集估计 Rx-modified ISQNN 参数：

```bash
cd "D:\研究生\研究生\Quantum Machine learning\code"
python -m Learning_Parity_with_Noise.estimate_theta_lpn_rx
```

成功后会生成：

```text
Learning_Parity_with_Noise/data/theta_lpn_rx.npz
```

统计每个 `theta_j` 有多少样本进入估计：

```bash
cd "D:\研究生\研究生\Quantum Machine learning\code"
python -m Learning_Parity_with_Noise.count_theta_estimation_samples
```

成功后会生成：

```text
Learning_Parity_with_Noise/data/theta_estimation_sample_counts.npz
Learning_Parity_with_Noise/data/theta_estimation_sample_counts.txt
```

利用已估计参数对测试集 `X_test` 做 ISQNN 推理：

```bash
cd "D:\研究生\研究生\Quantum Machine learning\code"
python -m Learning_Parity_with_Noise.generate_test_y_lpn
```

成功后会生成：

```text
Learning_Parity_with_Noise/data/lpn_test_predictions.npz
```

利用 Rx 参数对测试集 `X_test` 做 Rx-modified ISQNN 推理：

```bash
cd "D:\研究生\研究生\Quantum Machine learning\code"
python -m Learning_Parity_with_Noise.generate_test_y_lpn_rx
```

成功后会生成：

```text
Learning_Parity_with_Noise/data/lpn_test_predictions_rx.npz
```

比较 `Y_pred` 和底层 `S` 生成的 clean label：

```bash
cd "D:\研究生\研究生\Quantum Machine learning\code"
python -m Learning_Parity_with_Noise.evaluate_lpn_generation
```

成功后会生成：

```text
Learning_Parity_with_Noise/data/lpn_generation_metrics.npz
Learning_Parity_with_Noise/data/lpn_generation_metrics.txt
```

比较 Rx 推断的 `Y_pred` 和底层 `S` 生成的 clean label：

```bash
cd "D:\研究生\研究生\Quantum Machine learning\code"
python -m Learning_Parity_with_Noise.evaluate_lpn_generation_rx
```

成功后会生成：

```text
Learning_Parity_with_Noise/data/lpn_generation_metrics_rx.npz
Learning_Parity_with_Noise/data/lpn_generation_metrics_rx.txt
```

对指定测试样本做 majority vote 实验：

```bash
cd "D:\研究生\研究生\Quantum Machine learning\code"
python -m Learning_Parity_with_Noise.majority_vote_single_sample
```

查看保存内容可以用：

```python
import numpy as np

data = np.load("Learning_Parity_with_Noise/data/lpn_dataset.npz")
print(data.files)
print(data["X_train"].shape)
print(data["Y_train_noisy"].shape)
```

## 后续整体流程

当前已经完成：

```text
1. config.py
2. gf2_utils.py
3. test__LPN.py
4. generate_lpn_dataset.py
5. estimate_theta_lpn.py
6. count_theta_estimation_samples.py
7. generate_test_y_lpn.py
8. evaluate_lpn_generation.py
9. majority_vote_single_sample.py
10. estimate_theta_lpn_rx.py
11. generate_test_y_lpn_rx.py
12. evaluate_lpn_generation_rx.py
```

后续计划：

```text
1. run_pipeline.py
   一键执行：生成数据 -> 估计 theta -> 测试生成 -> 评估。
```

最终希望回答的问题是：

```text
ISQNN 是否能从有限 noisy LPN 样本中学习到 hidden parity map，
并在新输入 X_test 上生成接近 Y_test_clean 的输出？
```
