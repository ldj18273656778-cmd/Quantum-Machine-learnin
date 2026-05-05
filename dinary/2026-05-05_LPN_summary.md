# 2026-05-05 Learning Parity with Noise 项目总结

## 今日目标

今天从原来的 Rx-modified ISQNN/MNIST 问题切换到一个新方向：Learning Parity with Noise (LPN)。目标是在 `code/Learning_Parity_with_Noise/` 中建立一个矩阵版 LPN 项目，并尝试把 ISQNN / Rx-modified ISQNN 当作生成模型使用。

核心任务是：

```text
给定输入 x 和 hidden secret matrix S，生成 y = x S mod 2；
用有限训练集估计 ISQNN 参数 theta；
再对测试输入 X_test 进行生成推断；
最后比较生成结果 Y_pred 和真实 clean label Y_from_S。
```

## LPN Setup 确认

你明确了标准 LPN 的记号：

```text
x in {0,1}^{n_x}
s in {0,1}^{n_x}
y = <x, s> mod 2
```

为了生成多个输出 bit，将多个 secret vectors 作为列组成矩阵：

```text
S in {0,1}^{n_x x n_y}
y = x S mod 2
```

有限数据集写作：

```text
X in {0,1}^{num_samples x n_x}
Y_clean = X S mod 2
Y_noisy = Y_clean XOR E
E_ij ~ Bernoulli(noise_rate)
```

当前为了接入 ISQNN，采用直接映射：

```text
n_x = n_y = n1 * m
```

## 变量命名调整

最初使用了 `A`, `D`, `K` 等记号。后来按你的要求统一改成：

```text
X_train, X_test: 输入矩阵
Y_train_clean, Y_train_noisy: 训练标签
Y_test_clean, Y_test_noisy: 测试标签
S: hidden secret matrix
n_x: 输入 bit 数
n_y: 输出 bit 数
```

`config.py` 中也同步改成：

```text
n_x
n_y
n1
m
num_train
num_test
noise_rate
seed
```

## 已完成文件

主要目录：

```text
code/Learning_Parity_with_Noise/
```

今日新增或重写的主要脚本包括：

```text
config.py
gf2_utils.py
estimate_theta_lpn.py
count_theta_estimation_samples.py
generate_test_y_lpn.py
evaluate_lpn_generation.py
majority_vote_single_sample.py
estimate_theta_lpn_rx.py
generate_test_y_lpn_rx.py
evaluate_lpn_generation_rx.py
README.md
```

## 数据生成模块

`gf2_utils.py` 实现了 GF(2) 基础工具：

```text
sample_secret(n_x, n_y, rng)
sample_inputs(num_samples, n_x, rng)
gf2_matmul(x, s)
generate_labels(x, s)
add_bernoulli_noise(labels, noise_rate, rng)
generate_lpn_split(num_samples, s, noise_rate, rng)
bit_accuracy(predicted, target)
sample_accuracy(predicted, target)
hamming_distance(predicted, target)
```

`test__LPN.py` 用于小规模 sanity test，可以生成 `S`, `X`, `Y_clean`, `Y_noisy` 并打印样本。

`generate_lpn_dataset.py` 生成正式 train/test dataset，输出：

```text
code/Learning_Parity_with_Noise/data/lpn_dataset.npz
```

其中包括：

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
n1
m
n_bits
```

## 原始 ISQNN 参数估计与推断

`estimate_theta_lpn.py` 使用原始 ISQNN 学习方法估计参数。它引用：

```python
from sampling.ISQNN_generate_y import idqnn_connectivity
from Train.estimate_theta_from_filtered_samples import estimate_theta_from_filtered_samples
```

对每个 target bit `j`，筛选训练样本：

```text
x_j = 0
x_neighbors = 1
```

再用筛选后的 `y_j` 估计 `theta_j`。

输出：

```text
code/Learning_Parity_with_Noise/data/theta_lpn.npz
```

`count_theta_estimation_samples.py` 用于统计每个 `theta_j` 有多少训练样本进入估计集合，并用柱状图显示 `N_sp`。输出：

```text
data/theta_estimation_sample_counts.npz
data/theta_estimation_sample_counts.txt
```

`generate_test_y_lpn.py` 使用估计好的 `theta_lpn.npz`，对 `X_test` 逐样本调用原始 ISQNN 推断，并加入 `tqdm` 进度条。输出：

```text
data/lpn_test_predictions.npz
```

`evaluate_lpn_generation.py` 比较：

```text
Y_pred
Y_from_S = X_test S mod 2
```

评估指标包括：

```text
bit_accuracy
sample_accuracy
mean_hamming_distance
median_hamming_distance
max_hamming_distance
per_output_bit_accuracy
```

一次运行结果显示原始 ISQNN 基本接近随机：

```text
bit_accuracy: 0.498394
sample_accuracy: 0.000000
mean_hamming_distance: 8.025700
median_hamming_distance: 8.000000
max_hamming_distance: 15
```

## Majority Vote 实验

你提出同一个 `x_test` 多次采样是否能提高准确率。我们分析后认为：

```text
如果 P_theta(y_j = true_y_j | x_test) > 1/2，多次采样 + majority vote 可以提高准确率；
如果分布本身接近 1/2，多次采样也无法显著提高。
```

于是新增：

```text
majority_vote_single_sample.py
```

该脚本对指定测试样本重复运行 ISQNN `num_shots` 次，再逐 bit 多数投票。

默认实验：

```text
target_sample_idx = 0
num_shots = 11
```

一次结果为：

```text
vote bit accuracy: 0.437500
vote hamming distance: 9/16
mean single-shot bit accuracy: 0.488636
best single-shot bit accuracy: 0.625000
worst single-shot bit accuracy: 0.312500
```

说明对该样本，majority vote 没有改善结果。

## Rx-modified ISQNN 版本

你后来要求重新写一套使用 `Rx_modified` 的推断和参数估计脚本。我们保留原始 ISQNN 脚本不动，新增 Rx 版本。

`config.py` 中新增：

```text
THETA_RX_PATH = data/theta_lpn_rx.npz
PREDICTION_RX_PATH = data/lpn_test_predictions_rx.npz
rx_angle = pi / 4
```

新增脚本：

```text
estimate_theta_lpn_rx.py
generate_test_y_lpn_rx.py
evaluate_lpn_generation_rx.py
```

Rx 参数估计引用：

```python
from Rx_modified.ISQNN_generate_y_rx import idqnn_connectivity
from Rx_modified.estimate_theta_rx import estimate_theta_rx
```

Rx 推断引用：

```python
from Rx_modified.ISQNN_generate_y_rx import ISQNN_generate_y_rx
```

Rx 参数估计公式为：

```text
P(y_j = 1 | x_j = 0, x_neighbors = 1)
= 1/2 * [1 - (cos phi)^d * cos(theta_j)]
```

因此：

```text
theta_hat_j = arccos((1 - 2 p_hat) / (cos phi)^d)
```

其中：

```text
p_hat = mean(y_j)
phi = rx_angle
```

代码中等价写法为：

```python
alpha_d = -(cos(rx_angle)) ** d
arg = (2 * p_hat - 1) / alpha_d
theta_hat = arccos(arg)
```

Rx 参数估计已完整运行成功：

```text
estimated bits: 16/16
failed bits: []
saved: data/theta_lpn_rx.npz
```

Rx 推断脚本也通过前 1 个测试样本 smoke test，进度条正常。

## 今日关键观察

你发现加入 Rx 后结果似乎和不加 Rx 没有明显区别。我们分析原因是：

```text
LPN 的 y_j = <x, S[:, j]> mod 2 是全局 parity；
而当前 ISQNN / Rx-ISQNN 参数估计只使用局部条件统计量：
x_j = 0 且 x_neighbors = 1 下的 mean(y_j)。
```

对于随机 LPN secret matrix `S`，即使固定 `x_j` 和邻居，剩余未固定 bit 仍然会让 parity 近似均匀，因此：

```text
p_hat ≈ 1/2
```

原始公式得到：

```text
theta_hat ≈ arccos(0) = pi/2
```

Rx 公式中虽然多了：

```text
(cos phi)^d
```

但如果 `p_hat ≈ 1/2`，分子仍然接近 0，因此：

```text
theta_hat ≈ pi/2
```

所以加入 Rx 后，估计参数仍然大多在 `pi/2` 附近，推断结果仍接近随机。

## 当前运行方式

从 `code/` 目录运行：

```bash
python -m Learning_Parity_with_Noise.generate_lpn_dataset
python -m Learning_Parity_with_Noise.estimate_theta_lpn
python -m Learning_Parity_with_Noise.generate_test_y_lpn
python -m Learning_Parity_with_Noise.evaluate_lpn_generation
```

Rx 版本：

```bash
python -m Learning_Parity_with_Noise.estimate_theta_lpn_rx
python -m Learning_Parity_with_Noise.generate_test_y_lpn_rx
python -m Learning_Parity_with_Noise.evaluate_lpn_generation_rx
```

单样本 majority vote：

```bash
python -m Learning_Parity_with_Noise.majority_vote_single_sample
```

统计每个参数使用样本数：

```bash
python -m Learning_Parity_with_Noise.count_theta_estimation_samples
```

## 后续可能方向

后续如果继续推进，可以考虑：

```text
1. 减小 num_test，用小测试集快速比较原始 ISQNN 与 Rx-ISQNN。
2. 构造局部 secret matrix S，使 y_j 只依赖 j 和 neighbors(j)，验证 ISQNN 估计公式是否能成功。
3. 对随机 S 的 LPN，探索是否需要完全不同的学习方法，而不是局部 theta 估计公式。
4. 写 run_pipeline.py，一键完成 dataset -> theta -> inference -> evaluation。
5. 为 Rx 版本也写 single-sample majority vote 脚本。
```
