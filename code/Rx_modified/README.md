# Rx_modified

在 IDQNN/DQNN 电路中加入 Rx 旋转门，以打破 `|0⟩` 编码造成的 CZ 信息阻断，实现量子信息的全局传播。

---

## 目录结构

```
code/Rx_modified/
├── README.md                        ← 本文件
├── DQNN_generate_y_rx.py           ← 带 Rx 门的 DQNN 生成器
├── ISQNN_generate_y_rx.py          ← 带 Rx 门的 ISQNN 生成器
├── estimate_theta_rx.py            ← [新建] 适配 Rx 的 θ 估计公式
├── test_rx_effect.py                ← 3×3 网格快速验证
├── test_mnist_quick.py              ← MNIST 小批量快速推理
├── test_MNIST.py                    ← [修改] 使用 Rx 估计公式
├── MNIST_testdataset.py             ← [修改] 推理时使用 Rx
├── data/                            ← 输出数据（.npz）
└── output_images/                   ← 输出图像
```

**原则**：`code/sampling/`、`code/Train/`、`code/MNIST/` 中的原始代码完全未动。所有修改在 `Rx_modified/` 内独立完成。

---

## 数学原理

### 问题

原始电路编码规则：`x=0 → |+⟩`，`x=1 → |0⟩`。`|0⟩` 是 CZ 门的本征态（`CZ ⊗ |0⟩ = I`），连续 `|0⟩` 比特阻断 CZ 纠缠，将量子态切分成独立的连通分量。输出分布分解为独立块的乘积——无法产生全局图像。

### 方案

在编码之后、Rz(θ) 之前插入 `Rx(φ)` 门：

$$
R_x(\phi)|0\rangle = \cos(\phi/2)|0\rangle - i\sin(\phi/2)|1\rangle
$$

`|1⟩` 分量使 CZ 门能够施加 Z 操作，信息得以跨过原先阻断。

- 推理阶段：`φ = π/4`（所有比特），打破阻断
- 估计阶段：`φ_neighbor = 0`（邻居保持在 CZ 本征态），闭合估计公式不变：`θ̂ = arccos(1 - 2E[y])`

---

## 文件说明

### 1. `DQNN_generate_y_rx.py`（修改自 `code/sampling/DQNN_generate_y.py`）

```python
def DQNN_generate_y_rx(bitstring, n1, m, theta_list, rx_angle=0.0):
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `bitstring` | `str` | 长度为 `n1*m` 的二进制字符串，每位 `'0'` 或 `'1'` |
| `n1` | `int` | 层数（Block 数） |
| `m` | `int` | 每层量子比特数 |
| `theta_list` | `list[float]` | 旋转参数列表，长度 `n1*m`，也可传入标量（自动广播） |
| `rx_angle` | `float` | Rx 旋转角（弧度），默认 `0.0`（等价于原始行为） |

**返回值**：`(full_circuit, y)`
- `full_circuit`: `cirq.Circuit`，完整的量子电路
- `y`: `list[int]`，长度为 `n1*m` 的生成比特串

**修改点**（共 3 处）：
1. 初始编码后、Rz 前插入 Rx（第 34–37 行）
2. 每个 Block 的测量/重置后、Rz/CZ 前插入 Rx（第 116–120 行）

### 2. `ISQNN_generate_y_rx.py`（修改自 `code/sampling/ISQNN_generate_y.py`）

```python
def ISQNN_generate_y_rx(bitstring, n1, m, theta_list, rx_angle=0.0):
```

参数与返回值同上。

同时提供辅助函数：

```python
def idqnn_connectivity(n1, m):
```
**输入**：`n1` (层数), `m` (每层 qubit 数)
**输出**：`dict`，键为 `n1`, `m`, `n`, `intra_slice_edges`, `inter_slice_edges`, `all_edges`

**修改点**（1 处）：
- 所有 Slice 编码完成后、Rz/CZ 前插入 Rx（Step 1.5）

### 3. `estimate_theta_rx.py`（新建）

```python
def estimate_theta_rx(x, y, target_bit, adjacency, rx_angle=np.pi/4):
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `x` | `np.ndarray (N,)` | 输入比特串数组 |
| `y` | `np.ndarray (N, n)` | 目标输出数组 |
| `target_bit` | `int` | 目标比特索引（0-based） |
| `adjacency` | `list[set[int]]` | 邻接表 |
| `rx_angle` | `float` | Rx 旋转角（弧度），默认 `π/4` |

**返回值**：`dict`，键包含：
- `theta_hat_rad`, `theta_hat_deg`: 估计的 θ 值
- `N_sp`, `sum_y`, `p_hat`: 统计量
- `alpha_d`: 依赖邻居数的系数，α_d = -(cos φ)^d
- `n_neighbors`: CZ 邻居数 d

**估计公式**：  
$$P(y_j=1) = \frac{1 + \alpha_d \cos\theta_j}{2}, \quad \alpha_d = -(\cos\phi)^d$$
$$\hat{\theta}_j = \arccos\left(\frac{2\hat{P} - 1}{\alpha_d}\right)$$

当 φ=0 时 α_d=-1，退化为原公式 $\hat{\theta} = \arccos(1-2\bar{y})$。

### 4. `test_rx_effect.py`（新建）

3×3 网格快速验证脚本，对比 `rx_angle=0` 与 `rx_angle=π/4` 的输出分布。

**输入**：手动参数区设置 `n1`, `m`, `bitstring`, `theta`, `rx_test_angle`

**输出**（打印到控制台）：
- 每个量子比特的边际概率 `P(y_j=1)`
- 输出支持集大小（非零概率态数）
- 输出熵（bits）
- 行间平均相关系数（row0-row1, row0-row2, row1-row2）
- 边际概率最大变化量

```python
def compute_marginals(probs, n):
    """从全概率分布提取每比特 P(y_j=1)。"""
# 输入: probs (array[2^n]), n (int)
# 输出: margins (array[n])

def compute_pairwise(probs, n):
    """提取边际 + 两两相关系数矩阵。"""
# 输入: probs, n
# 输出: (margins, corr_matrix)

def analyze_circuit(n1, m, bitstring, theta_list, rx_angle):
    """运行 ISQNN 电路，返回全概率分布和电路对象。"""
# 输入: n1, m, bitstring, theta_list, rx_angle
# 输出: (probs_array, cirq_circuit)
```

### 5. `test_mnist_quick.py`（新建）

**作用**：用小批量 MNIST 测试样本对比无 Rx 与 Rx(π/4) 的推理质量。

**输入**（手动参数区）：
- `NUM_TEST_SAMPLES`: 测试样本数（默认 100）
- `rx_angle`: Rx 角（默认 π/4）
- `threshold`: MNIST 二值化阈值
- `n1`, `m`: 模型维度

**数据输入**：
- `test_MNIST_10x10_binarize0.5.npy`：MNIST 测试图像
- `test_MNIST_labels.npy`：标签
- `estimate_theta_binarized0.4.npz`：预估的 θ 参数

**输出**（保存到 `data/mnist_quick_rx_pi4_N{N}.npz`）：
- `y_no_rx`: 无 Rx 的推理输出
- `y_with_rx`: Rx(π/4) 的推理输出
- `y_target`: 目标标签编码
- 各项评估指标准确率、Hamming 距离等

### 6. `test_MNIST.py`（修改自 `code/MNIST/test_MNIST.py`）

**作用**：从 MNIST 训练数据中估计 θ 参数。

**修改**：
- 使用本地 `estimate_theta_rx()` 替代原来的 `estimate_theta_from_filtered_samples()`
- 估计公式适配 Rx：`θ̂ = arccos((2P-1)/α_d)`，其中 α_d = -(cos φ)^d
- `rx_est_angle = π/4` 作为估计时使用的 Rx 角
- 输出到 `RX_DIR / "data" / estimate_theta_rx_pi4_binarized{threshold}.npz`

**输出**：`data/estimate_theta_binarized{threshold}.npz`

包含：`theta_hat_flat` (100,), `theta_hat_matrix` (10×10), `skipped_bits`

### 7. `MNIST_testdataset.py`（修改自 `code/MNIST/MNIST_testdataset.py`）

**作用**：用估计的 θ 参数进行 MNIST 推理。

**修改**：
- 导入本地 `from DQNN_generate_y_rx import DQNN_generate_y_rx`
- `MNIST_DIR` 指向原始 `code/MNIST/`，`RX_DIR` 指向 `code/Rx_modified/`
- 调用 `DQNN_generate_y_rx(..., rx_angle=rx_test_angle)`（默认 π/4）
- 输出到 `RX_DIR / "data" /`

---

## 指标解释

### Bit-level accuracy（比特级准确率）

```
acc = (预测 y 与目标 y 完全一致的比特数) / 总比特数
```

取值范围 [0, 1]。随机猜测 = 50%。衡量每个像素级别的预测正确率。

### Hamming distance（汉明距离）

```
d_H(y_pred, y_target) = (不同比特的数量) / 总比特数
```

取值范围 [0, 1]。越小越好（0 = 完美匹配）。等价于 `1 - accuracy`。

### Marginal probability（边际概率）

```
P(y_j = 1) = 在多次采样中第 j 个量子比特输出为 1 的比例
```

N 个样本取平均。若所有边际概率均为 0.5，说明模型无法区分该比特倾向于输出 0 还是 1。

#### |dP| per qubit

```
|dP|_j = |P_rx(y_j=1) - P_no_rx(y_j=1)|
```

衡量 Rx 门对每个比特边际概率的改变幅度。Max = 最大变化，Mean = 平均变化。

### Output support（输出支持集）

```
support = 概率非零的输出态数量
```

全 Hilbert 空间大小为 `2^n`。若 support = `2^n`，说明电路能触达所有可能的输出态。

### Entropy（输出熵）

```
H = -Σ_y P(y) · log₂ P(y)
```

最大值为 `n` bits（均匀分布）。衡量输出分布的不确定性。

### Correlation（相关系数）

```
corr(i, j) = E[y_i · y_j] − E[y_i] · E[y_j]
```

或使用 Pearson 相关系数：

```
ρ(i, j) = Cov(y_i, y_j) / √(Var(y_i) · Var(y_j))
```

衡量两个量子比特输出之间的统计依赖关系。正值表示倾向同向，负值表示倾向反向，0 表示统计独立。

#### Avg |corr_diff|

```
|corr_diff| = 平均绝对差 |ρ_rx(i,j) − ρ_no_rx(i,j)|
```

衡量 Rx 对关联结构的改变程度。

#### Avg |corr to target|

```
|corr to target| = 平均绝对差 |ρ_model(i,j) − ρ_target(i,j)|
```

衡量模型的关联结构与目标分布的关联结构的接近程度。越小越好。

---

## 使用流程

```bash
# 1. 快速验证 Rx 在 3×3 网格的效果
python code/Rx_modified/test_rx_effect.py

# 2. 估计 θ（若需要重新估计，输出到 Rx_modified/data/）
python code/Rx_modified/test_MNIST.py

# 3. MNIST 小批量快速推理（100 样本，约 30 秒）
python code/Rx_modified/test_mnist_quick.py

# 4. MNIST 全量推理（10000 样本，约 20 分钟）
#    编辑 MNIST_testdataset.py 中的 rx_test_angle，然后运行
python code/Rx_modified/MNIST_testdataset.py
```

---

## 当前测试结果（100 样本，Rx=π/4）

| 指标 | 无 Rx | Rx(π/4) | 解读 |
|------|-------|---------|------|
| Bit-level accuracy | 49.93% | **54.73%** | 超越随机 |
| Hamming distance | 0.501 | **0.453** | 距离减小 |
| Max |dP| per qubit | — | 0.270 | Rx 改变了边际 |
| Mean |dP| per qubit | — | 0.095 | 整体边际偏移 |
| Avg correlation to target | 0.203 | 0.203 | 关联结构未改善 |

**结论**：Rx 改善了边际层面的比特控制（+4.8% 准确率），但未能改善两两关联结构。100 个 θ 参数 + 固定 CZ 网格是本模型的表达能力瓶颈。
