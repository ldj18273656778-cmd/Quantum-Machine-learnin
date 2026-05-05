# Train

参数估计与数据生成模块。基于 ISQNN 模型采样生成的 $(x, y)^N$ 数据集，对图中每个量子比特的旋转参数 $\theta$ 进行估计，并评估估计精度。

## 目录结构

```
Train/
├── __init__.py                                # 包标记
├── generate_theta_demo.py                     # 生成随机 theta 参数矩阵
├── generate_Dx.py                             # 按论文分布 D(x) 采样输入 bitstring x
├── generate_xy_dataset.py                     # 生成 (x, y)^N 数据集
├── find_x_indices_by_graph_condition.py       # 按图邻域条件筛选样本索引
├── estimate_theta_from_filtered_samples.py    # 从筛选样本估计 θ̂
├── compare_theta_true_and_estimated.py        # 比较真实 θ 与估计 θ̂
├── test_train.py                              # 端到端集成测试
├── yinyong_test.py                            # 跨模块引用测试
├── test.ipynb                                 # Jupyter 交互式测试
├── README.md                                  # 本文档
└── data/                                      # 输入 / 输出数据文件 (.npy, .npz, .txt)
```

## 数据流概览

```
generate_theta_demo.py ──→ data/theta_demo.npy        (真实参数 θ)
         │
         ▼
generate_Dx.py ──→ 采样 D(x) 分布                     (输入 bitstring x)
         │
         ▼
generate_xy_dataset.py ──→ data/xy_dataset.npy         (x, y, comps, θ)
         │
         ▼
find_x_indices_by_graph_condition.py                    (筛选满足条件的索引)
         │
         ▼
estimate_theta_from_filtered_samples.py ──→ θ̂          (参数估计)
         │
         ▼
compare_theta_true_and_estimated.py                     (θ vs θ̂ 比较)
```

---

## 各文件功能与可调用函数详解

### 1. `generate_theta_demo.py` — 生成演示 theta 参数

生成一个 $(n_1 \times m)$ 的二维 theta 参数矩阵，保存为 `.npy` 文件。

**无对外可调用函数**（纯脚本）。

---

### 2. `generate_Dx.py` — 按分布 D(x) 采样输入

实现论文 arXiv:2509.09033 Appendix C.2.a 的输入分布 D(x)：

| 分量 | 概率 | 规则 |
|---|---|---|
| 0 | 1/3 | $x = 0^n$（全零） |
| 1 | 1/3 | 每比特 i.i.d., $P(0)=0.6,\;P(1)=0.4$ |
| 2 | 1/3 | 每比特 i.i.d., $P(0)=0.2,\;P(1)=0.8$ |

#### `sample_dx(n_bits, num_samples, seed) -> (x, comps)`

| 参数 | 类型 | 说明 |
|---|---|---|
| `n_bits` | `int` | 每条 bitstring 的位数 |
| `num_samples` | `int` | 采样数量 |
| `seed` | `int` | 随机种子 |

| 返回值 | 类型 | 说明 |
|---|---|---|
| `x` | `np.ndarray` | 形状 `(num_samples,)`，dtype `<U{n_bits}`，bitstring 数组 |
| `comps` | `np.ndarray` | 形状 `(num_samples,)`，dtype `int8`，分量编号 ∈{0,1,2} |

#### `generate_Dx(n, num_samples=10000, seed=7) -> x`

| 参数 | 类型 | 说明 |
|---|---|---|
| `n` | `int` | 每条 bitstring 的位数 |
| `num_samples` | `int` | 采样数量，默认 10000 |
| `seed` | `int` | 随机种子，默认 7 |

| 返回值 | 类型 | 说明 |
|---|---|---|
| `x` | `np.ndarray` | 形状 `(num_samples,)` 的 bitstring 数组 |

#### `save_comps_x(path, x, comps) -> path`

| 参数 | 类型 | 说明 |
|---|---|---|
| `path` | `str \| Path` | 输出文件路径 |
| `x` | `np.ndarray` | bitstring 数组 |
| `comps` | `np.ndarray` | 分量编号数组 |

| 返回值 | 类型 | 说明 |
|---|---|---|
| `path` | `Path` | 写入的文件路径（文本格式 `comp,x`） |

---

### 3. `generate_xy_dataset.py` — 生成 (x, y)^N 数据集

给定固定 theta 和 D(x) 分布，调用 ISQNN 生成对应的 y 值，保存为 `.npy` 和 `.txt`。

#### `load_theta(theta_path, n1, m) -> theta`

| 参数 | 类型 | 说明 |
|---|---|---|
| `theta_path` | `Path` | `.npy` 文件路径 |
| `n1` | `int` | 层数 |
| `m` | `int` | 每层量子比特数 |

| 返回值 | 类型 | 说明 |
|---|---|---|
| `theta` | `np.ndarray` | 形状 `(n1, m)`，dtype `float` |

#### `generate_xy_dataset(n1, m, theta, num_samples=1000, seed=7, show_progress=True) -> (x, y, comps)`

| 参数 | 类型 | 说明 |
|---|---|---|
| `n1` | `int` | 层数 |
| `m` | `int` | 每层量子比特数 |
| `theta` | `np.ndarray` | 形状 `(n1, m)` 的参数矩阵 |
| `num_samples` | `int` | 采样数量，默认 1000 |
| `seed` | `int` | 随机种子，默认 7 |
| `show_progress` | `bool` | 是否显示进度条 |

| 返回值 | 类型 | 说明 |
|---|---|---|
| `x` | `np.ndarray` | 形状 `(N,)`，dtype `str` 的 bitstring 数组 |
| `y` | `np.ndarray` | 形状 `(N, n1*m)`，dtype `int8` 的 y 比特数组 |
| `comps` | `np.ndarray` | 形状 `(N,)`，D(x) 分量编号 |

---

### 4. `find_x_indices_by_graph_condition.py` — 按图邻域条件筛选

对每个目标比特 j，筛选满足条件的样本索引：
1. $x_j = 0$
2. 对图中 j 的所有邻居 k，均有 $x_k = 1$

#### `load_x_bitstrings(path) -> x`

| 参数 | 类型 | 说明 |
|---|---|---|
| `path` | `Path` | `.npy`（含键 `x` 的字典）或 `.txt` / `.csv` 文件路径 |

| 返回值 | 类型 | 说明 |
|---|---|---|
| `x` | `np.ndarray` | 形状 `(N,)`，dtype `str` 的 bitstring 数组 |

#### `build_adjacency(n, edges) -> adj`

| 参数 | 类型 | 说明 |
|---|---|---|
| `n` | `int` | 节点总数 |
| `edges` | `Iterable[tuple[int,int]]` | 边列表，每个元素为 `(a, b)` |

| 返回值 | 类型 | 说明 |
|---|---|---|
| `adj` | `list[set[int]]` | 长度为 n 的邻接表，每个元素为对应节点的邻居集合 |

#### `find_indices(x, target_bit, adjacency, show_progress=True) -> indices`

| 参数 | 类型 | 说明 |
|---|---|---|
| `x` | `np.ndarray` | 形状 `(N,)` 的 bitstring 数组 |
| `target_bit` | `int` | 目标比特索引（0-based） |
| `adjacency` | `list[set[int]]` | 图邻接表 |
| `show_progress` | `bool` | 是否显示进度条 |

| 返回值 | 类型 | 说明 |
|---|---|---|
| `indices` | `np.ndarray` | 形状 `(K,)`，dtype `int`，满足条件的样本索引 |

---

### 5. `estimate_theta_from_filtered_samples.py` — 从筛选样本估计参数

核心公式（论文正文）：
$$\hat{\theta}_j = \arccos\!\left(1 - \frac{2}{N_{sp}} \sum_{t=1}^{N_{sp}} y_t\right)$$

#### `load_xy_from_npy(path) -> (x, y)`

| 参数 | 类型 | 说明 |
|---|---|---|
| `path` | `Path` | `.npy` 文件路径，包含键 `x` 和 `y` 的字典 |

| 返回值 | 类型 | 说明 |
|---|---|---|
| `x` | `np.ndarray` | 形状 `(N,)`，dtype `str` 的 bitstring 数组 |
| `y` | `np.ndarray` | 形状 `(N, n)`，dtype 依原始数据 |

#### `estimate_theta_from_filtered_samples(x, y, target_bit, adjacency, show_progress=True) -> result`

| 参数 | 类型 | 说明 |
|---|---|---|
| `x` | `np.ndarray` | 形状 `(N,)` 的 bitstring 数组 |
| `y` | `np.ndarray` | 形状 `(N, n)` 的 y 数据 |
| `target_bit` | `int` | 目标比特索引（0-based） |
| `adjacency` | `list[set[int]]` | 图邻接表 |
| `show_progress` | `bool` | 是否显示进度条 |

| 返回值 | 类型 | 说明 |
|---|---|---|
| `result` | `dict` | 包含键：`target_bit_0based`, `N_sp`（有效样本数）, `sum_y`（y 之和）, `theta_hat_rad`（弧度）, `theta_hat_deg`（度）, `indices_0based`（匹配的样本索引） |

---

### 6. `compare_theta_true_and_estimated.py` — 比较真实 θ 与估计 θ̂

计算 $\cos(\hat{\theta}) - \cos(\theta)$ 的误差统计。

#### `load_true_theta(path) -> theta`

| 参数 | 类型 | 说明 |
|---|---|---|
| `path` | `Path` | `.npy` 文件路径（含键 `theta` 的字典或直接为二维数组） |

| 返回值 | 类型 | 说明 |
|---|---|---|
| `theta` | `np.ndarray` | 形状 `(n1, m)`，真实 theta 参数 |

#### `load_estimated_theta(path) -> theta_hat`

| 参数 | 类型 | 说明 |
|---|---|---|
| `path` | `Path` | `.npy` 文件路径（含键 `theta_hat_matrix_rad` 的字典） |

| 返回值 | 类型 | 说明 |
|---|---|---|
| `theta_hat` | `np.ndarray` | 形状 `(n1, m)`，估计 theta 参数（弧度） |

#### `compare_theta(theta_true, theta_hat) -> result`

| 参数 | 类型 | 说明 |
|---|---|---|
| `theta_true` | `np.ndarray` | 形状 `(n1, m)`，真实参数 |
| `theta_hat` | `np.ndarray` | 形状 `(n1, m)`，估计参数 |

| 返回值 | 类型 | 说明 |
|---|---|---|
| `result` | `dict` | 包含键：`theta_true`, `theta_hat`, `d_cos`（$\cos\hat{\theta}-\cos\theta$）、`abs_err`（绝对误差）、`mae`（平均绝对误差）、`rmse`（均方根误差）、`max_abs`（最大绝对误差） |

---

### 7. `test_train.py` — 端到端集成测试

串联完整数据流：加载 theta → 加载 xy 数据集 → 逐比特估计 θ̂ → 与真实 θ 比较。

**无对外可调用函数**（`main() -> int` 为脚本内部函数）。

---

### 8. `yinyong_test.py` — 跨模块引用测试

验证 Train 模块能正确从 sampling 模块导入 DQNN/ISQNN 生成器并调用。

**无对外可调用函数**。

---

### 9. `data/` — 数据目录

| 文件 | 内容 |
|---|---|
| `theta_demo.npy` | 随机生成的 theta 参数矩阵 (n1×m) |
| `xy_dataset*.npy/.txt` | (x, y)^N 数据集，`.npy` 为字典格式，`.txt` 为文本格式 |
| `dx_comps_x_demo*.txt` | D(x) 采样结果导出 |
| `matched_indices.txt` | 筛选出的满足条件的样本索引 |
| `theta_estimate_all_bits.npy` | 所有比特的 θ̂ 估计结果字典 |

---

## 使用方法

```bash
# 1) 生成 theta 参数
python code/Train/generate_theta_demo.py

# 2) 生成 (x, y)^N 数据集
python code/Train/generate_xy_dataset.py

# 3) 筛选样本 & 估计 θ̂（对指定目标比特）
python code/Train/estimate_theta_from_filtered_samples.py

# 4) 比较 θ 与 θ̂
python code/Train/compare_theta_true_and_estimated.py

# 5) 端到端测试
python code/Train/test_train.py
```

作为模块导入：

```python
from Train.generate_Dx import sample_dx, generate_Dx
from Train.generate_xy_dataset import generate_xy_dataset, load_theta
from Train.find_x_indices_by_graph_condition import build_adjacency, find_indices, load_x_bitstrings
from Train.estimate_theta_from_filtered_samples import estimate_theta_from_filtered_samples, load_xy_from_npy
from Train.compare_theta_true_and_estimated import compare_theta
```

## 依赖

- **numpy** — 数值计算
- **cirq** — 量子电路模拟（通过 sampling 模块间接依赖）
- **tqdm** — 进度条（可选）
