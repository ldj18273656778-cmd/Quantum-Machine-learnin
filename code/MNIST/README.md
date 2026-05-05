# MNIST

10×10 二值化 MNIST 数据处理模块。包含标签编码、参数估计、推断生成、距离计算及网络连通性可视化。

## 目录结构

```
MNIST/
├── Encode_0to9.py                                # 数字 0-9 对角线编码
├── test_MNIST.py                                  # 训练集逐 bit 估计 θ
├── MNIST_testdataset.py                           # 测试集推断 y 生成
├── distance.py                                    # 样本与标准编码的欧氏距离
├── compare.py                                     # 推断结果 y_inferred vs y_encoded 可视化对比
├── connectivity_of_testbitstring.py               # 单样本 0-bit ISQNN 连通分量可视化
├── connectivity_of_testbitstring_with_label_diagonal.py  # 连通性 + 标签对角线叠加
├── plot_idqnn_network_connectivity.py             # ISQNN 网络结构可视化
├── compare.ipynb / draw some figure.ipynb         # Jupyter 交互式笔记本
├── load_mnist&test.ipynb                          # 数据加载与测试笔记本
├── README.md                                      # 本文档
├── data/                                          # 输入/输出数据文件
└── output_images/                                 # 生成的可视化图片
```

---

## 核心概念

- **10×10 二值化 MNIST**：原始 28×28 灰度图像下采样为 10×10 并二值化（threshold 阈值）
- **X_flip**：`1 - X`，将像素值 0/1 反转，使黑色笔画对应 bit=0
- **对角线编码**：数字 k ∈ {0,...,9} 编码为 10×10 矩阵，第 i 行第 `(i+k) mod 10` 列为 1
- **参数估计**：将 MNIST 图像视为 ISQNN 的输入 x，将编码标签视为 y，逐 bit 估计 θ

---

## 各文件功能与可调用函数详解

### 1. `Encode_0to9.py` — 数字 0-9 对角线编码

#### `encode_diagonal(k, n=10) -> np.ndarray`

| 参数 | 类型 | 说明 |
|---|---|---|
| `k` | `int` | 数字标签，0 ~ n-1 |
| `n` | `int` | 矩阵大小，默认 10 |

| 返回值 | 类型 | 说明 |
|---|---|---|
| `M` | `np.ndarray` | 形状 `(n, n)`，dtype `int`。第 i 行第 `(i+k) mod n` 列为 1，其余为 0 |

**示例**：`encode_diagonal(0)` 返回主对角线为 1 的矩阵，`encode_diagonal(1)` 右移一位。

---

### 2. `test_MNIST.py` — 训练集逐 bit 参数估计

主脚本。读取 MNIST 训练集（二值化图像 X 和标签 y），对每个目标比特按 ISQNN 图邻域条件筛选满足 `x_j=0` 且邻居全为 1 的样本，调用 `estimate_theta_from_filtered_samples` 估计 θ̂，最终保存为 `.npz`。

**无对外可调用函数**（纯脚本）。

输出文件 `data/estimate_theta_binarized{threshold}.npz` 包含键：
- `theta_hat_flat` — 估计参数一维数组
- `theta_hat_matrix` — 形状 (n1, m) 的参数矩阵
- `skipped_bits` — N_sp=0 跳过的比特索引

---

### 3. `MNIST_testdataset.py` — 测试集推断 y 生成

主脚本。加载估计的 θ̂ 参数矩阵，读取测试集，对每个测试样本调用 `DQNN_generate_y` 生成推断输出 `y_inferred`，同时计算 `y_test_encoded`（测试标签的对角线编码），保存为 `.npz`。

**无对外可调用函数**（纯脚本）。

输出文件 `data/y_inferred_binarized{threshold}.npz` 包含键：
- `y_inferred` — 推断输出数组
- `y_test_encoded` — 标签对角线编码数组

---

### 4. `distance.py` — 欧氏距离计算与可视化

#### `distance(x, y) -> float`

| 参数 | 类型 | 说明 |
|---|---|---|
| `x` | `np.ndarray` | 第一个数组（任意形状，内部 ravel） |
| `y` | `np.ndarray` | 第二个数组（必须与 x 同形状） |

| 返回值 | 类型 | 说明 |
|---|---|---|
| `dist` | `float` | 欧氏距离 ∥x − y∥₂ |

#### `build_distance_table(target_grid) -> pd.DataFrame`

| 参数 | 类型 | 说明 |
|---|---|---|
| `target_grid` | `np.ndarray` | 形状 (n, n) 的目标矩阵（如测试样本或生成均值） |

| 返回值 | 类型 | 说明 |
|---|---|---|
| `df` | `pd.DataFrame` | 含 `Rank`、`Digit`、`Distance` 列，按距离升序排列（距离取自 target_grid 与各数字对角线编码之间的欧氏距离） |

#### `load_test_sample(sample_index, threshold, n1, m) -> (sample_grid, true_label, test_path)`

| 参数 | 类型 | 说明 |
|---|---|---|
| `sample_index` | `int` | 测试样本索引 |
| `threshold` | `float` | 二值化阈值 |
| `n1` | `int` | 行数 |
| `m` | `int` | 列数 |

| 返回值 | 类型 | 说明 |
|---|---|---|
| `sample_grid` | `np.ndarray` | 形状 (n1, m)，该样本的图像数据 |
| `true_label` | `int` | 该样本的真实标签 |
| `test_path` | `Path` | 加载的测试文件路径 |

#### `load_generated_mean(sample_index, threshold, n1, m, num_trials) -> (generated_mean_grid, generated_path)`

| 参数 | 类型 | 说明 |
|---|---|---|
| `sample_index` | `int` | 样本索引 |
| `threshold` | `float` | 二值化阈值 |
| `n1` | `int` | 行数 |
| `m` | `int` | 列数 |
| `num_trials` | `int` | 生成时使用的重复试验次数 |

| 返回值 | 类型 | 说明 |
|---|---|---|
| `generated_mean_grid` | `np.ndarray` | 形状 (n1, m)，多次生成的平均输出 |
| `generated_path` | `Path` | 加载的文件路径 |

#### `get_distance_target(source_mode, sample_grid, generated_mean_grid) -> (distance_grid, distance_label)`

| 参数 | 类型 | 说明 |
|---|---|---|
| `source_mode` | `str` | `"test_image"` 或 `"generated_mean"` |
| `sample_grid` | `np.ndarray` | 测试样本图像 |
| `generated_mean_grid` | `np.ndarray` 或 `None` | 生成均值（source_mode=`"generated_mean"` 时必需） |

| 返回值 | 类型 | 说明 |
|---|---|---|
| `distance_grid` | `np.ndarray` | 用于计算距离的目标矩阵 |
| `distance_label` | `str` | 描述文字（如 `"Selected test image"`） |

#### `plot_distance_bars(distance_label, sample_index, true_label, num_trials, distance_table, output_path) -> None`

| 参数 | 类型 | 说明 |
|---|---|---|
| `distance_label` | `str` | 距离来源描述 |
| `sample_index` | `int` | 样本索引 |
| `true_label` | `int` | 真实标签 |
| `num_trials` | `int` | 试验次数 |
| `distance_table` | `pd.DataFrame` | `build_distance_table` 输出的表格 |
| `output_path` | `Path` | 图片保存路径 |

绘制水平条形图：x 轴为欧氏距离，y 轴为 digit 0-9，绿色标记真实标签即为最近、红色标记真实标签非最近、蓝色标记非真实标签最近。

---

### 5. `compare.py` — 推断与编码可视化对比

加载 `y_inferred.npz`，将 `y_inferred` 和 `y_test_encoded` 各取前 10 个样本 reshape 为 10×10 图像并排显示。

**无对外可调用函数**（纯脚本）。

---

### 6. `connectivity_of_testbitstring.py` — 单样本 0-bit 连通分量可视化

#### `build_adjacency(n, edges) -> list[set[int]]`

| 参数 | 类型 | 说明 |
|---|---|---|
| `n` | `int` | 节点总数 |
| `edges` | `list[tuple[int,int]]` | 边列表 |

| 返回值 | 类型 | 说明 |
|---|---|---|
| `adjacency` | `list[set[int]]` | 长度为 n 的邻接表 |

#### `get_selected_components(bit_grid, edges, selected_bit_value) -> (selected_nodes, selected_edges, components)`

| 参数 | 类型 | 说明 |
|---|---|---|
| `bit_grid` | `np.ndarray` | 形状 (n1, m) 的比特矩阵 |
| `edges` | `list[tuple[int,int]]` | ISQNN 全部边 |
| `selected_bit_value` | `int` | 目标比特值（0 或 1） |

| 返回值 | 类型 | 说明 |
|---|---|---|
| `selected_nodes` | `list[int]` | 值为 selected_bit_value 的比特索引 |
| `selected_edges` | `list[tuple[int,int]]` | 两端都在 selected_nodes 中的边 |
| `components` | `list[list[int]]` | 连通分量列表，按大小降序排列 |

#### `bit_to_coord(bit_index, width) -> tuple[int, int]`

| 参数 | 类型 | 说明 |
|---|---|---|
| `bit_index` | `int` | 展平索引 |
| `width` | `int` | 每行宽度（即 m） |

| 返回值 | 类型 | 说明 |
|---|---|---|
| `(row, col)` | `tuple[int,int]` | 行列坐标 |

#### `print_component_report(components, width, selected_bit_value) -> None`

打印连通分量的详细信息（索引、坐标），每行一个分量。

#### `style_grid_axis(ax, rows, cols) -> None`

为 matplotlib Axes 配置网格样式（灰色细线、等比例）。

#### `plot_connectivity(bit_grid, label, components, selected_edges, selected_bit_value, output_path) -> None`

| 参数 | 类型 | 说明 |
|---|---|---|
| `bit_grid` | `np.ndarray` | 图像矩阵 |
| `label` | `int` | 样本标签 |
| `components` | `list[list[int]]` | 连通分量 |
| `selected_edges` | `list[tuple[int,int]]` | 被选中的边 |
| `selected_bit_value` | `int` | 目标比特值 |
| `output_path` | `Path` | 保存路径 |

绘制双栏图：左栏为原始图像，右栏叠加 ISQNN 连通边与连通分量（不同分量不同颜色），保存为 PNG。

---

### 7. `connectivity_of_testbitstring_with_label_diagonal.py` — 连通性 + 标签对角线叠加

在 connectivity 基础上叠加 `encode_diagonal(label)` 参考线。

#### `draw_connectivity(ax, bit_grid, components, selected_edges) -> None`

| 参数 | 类型 | 说明 |
|---|---|---|
| `ax` | `plt.Axes` | 目标绘图轴 |
| `bit_grid` | `np.ndarray` | 图像矩阵 |
| `components` | `list[list[int]]` | 连通分量 |
| `selected_edges` | `list[tuple[int,int]]` | 被选中的边 |

在指定 Axes 上绘制半透明背景 + 连通边 + 分量散点。

#### `overlay_reference_diagonal(ax, reference_grid, label_value) -> list[int]`

| 参数 | 类型 | 说明 |
|---|---|---|
| `ax` | `plt.Axes` | 目标绘图轴 |
| `reference_grid` | `np.ndarray` | 对角线编码矩阵 |
| `label_value` | `int` | 标签值（图例用） |

| 返回值 | 类型 | 说明 |
|---|---|---|
| `diagonal_indices` | `list[int]` | 对角线编码中值为 1 的比特索引（展平编号） |

在图上叠加红色方框 + X 标记。

#### `print_diagonal_report(diagonal_indices, overlap_indices, width, label_value, selected_bit_value_local) -> None`

打印对角线编码与当前 0-bit 的重叠信息。

#### `plot_connectivity_with_reference_diagonal(bit_grid, sample_index, sample_label, components, selected_edges, selected_nodes, reference_grid, output_path) -> (diagonal_indices, overlap_indices)`

| 参数 | 类型 | 说明 |
|---|---|---|
| `bit_grid` | `np.ndarray` | 形状 (n1, m) 的图像矩阵 |
| `sample_index` | `int` | 样本索引 |
| `sample_label` | `int` | 样本真实标签 |
| `components` | `list[list[int]]` | 连通分量 |
| `selected_edges` | `list[tuple[int,int]]` | 被选中边 |
| `selected_nodes` | `list[int]` | 被选中节点 |
| `reference_grid` | `np.ndarray` | 对角线编码矩阵 |
| `output_path` | `Path` | 图片保存路径 |

| 返回值 | 类型 | 说明 |
|---|---|---|
| `diagonal_indices` | `list[int]` | 对角线 1-bit 索引 |
| `overlap_indices` | `list[int]` | 对角线与 0-bit 的重叠索引 |

绘制三栏图：原始图像 / 连通性图 / 连通+对角线叠加图。

---

### 8. `plot_idqnn_network_connectivity.py` — ISQNN 网络结构可视化

完全独立于 MNIST 数据，仅可视化 ISQNN 网络的连接拓扑。

#### `ensure_interactive_backend() -> None`

检测并切换 matploblib 后端为交互式（qtagg/tkagg），避免在无 GUI 环境下无法 `plt.show()`。

#### `display_index_to_coord(bit_index, m) -> tuple[int, int]`

| 参数 | 类型 | 说明 |
|---|---|---|
| `bit_index` | `int` | 展平索引 |
| `m` | `int` | 每行宽度 |

| 返回值 | 类型 | 说明 |
|---|---|---|
| `(row, col)` | `tuple[int,int]` | 行列坐标 |

#### `style_axis(ax, n1, m) -> None`

配置 Axes 的网格、刻度、范围、等比例。

#### `draw_edges(ax, edges, m, color, linewidth) -> None`

| 参数 | 类型 | 说明 |
|---|---|---|
| `ax` | `plt.Axes` | 目标轴 |
| `edges` | `list[tuple[int,int]]` | 边列表 |
| `m` | `int` | 每行宽度 |
| `color` | `str` | 线条颜色 |
| `linewidth` | `float` | 线宽 |

#### `annotate_nodes(ax, n1, m) -> None`

在每个节点位置标注其 0-based bit 编号（白底黑字圆角框）。

#### `plot_network_connectivity(n1, m) -> None`

| 参数 | 类型 | 说明 |
|---|---|---|
| `n1` | `int` | 层数（行数） |
| `m` | `int` | 每层比特数（列数） |

绘制整张网络图：蓝色边 = intra-slice（层内），橙色边 = inter-slice（层间），直接 `plt.show()` 显示。

---

### 9. Notebooks

| 文件 | 说明 |
|---|---|
| `compare.ipynb` | 交互式推断与编码对比分析 |
| `draw some figure.ipynb` | 通用绘图笔记本 |
| `load_mnist&test.ipynb` | MNIST 数据加载与测试 |

---

### 10. `data/` — 数据目录

| 文件 | 内容 |
|---|---|
| `MNIST_10x10_binarize{threshold}.npy` | 训练集二值化图像 |
| `test_MNIST_10x10_binarize{threshold}.npy` | 测试集二值化图像 |
| `MNIST_labels.npy` / `test_MNIST_labels.npy` | 训练/测试标签 |
| `estimate_theta_binarized{threshold}.npz` | 估计的 θ 参数 |
| `y_inferred_binarized{threshold}.npz` | 测试集推断 y 与编码 y_test |

## 使用方法

```bash
# 参数估计
python code/MNIST/test_MNIST.py

# 测试集推断
python code/MNIST/MNIST_testdataset.py

# 推断结果可视化
python code/MNIST/compare.py

# ISQNN 网络结构可视化
python code/MNIST/plot_idqnn_network_connectivity.py

# 单样本连通性可视化
python code/MNIST/connectivity_of_testbitstring.py

# 连通性 + 对角线叠加
python code/MNIST/connectivity_of_testbitstring_with_label_diagonal.py

# 距离计算与对比
python code/MNIST/distance.py
```

作为模块导入：

```python
from MNIST.Encode_0to9 import encode_diagonal
from MNIST.distance import distance, build_distance_table
from MNIST.connectivity_of_testbitstring import get_selected_components, bit_to_coord

M = encode_diagonal(3)                     # 3 的对角线编码
d = distance(M, encode_diagonal(5))         # 编码间欧氏距离
```

## 依赖

- **numpy** — 数值计算
- **matplotlib** — 可视化
- **pandas** — distance.py 表格输出
- **cirq** — 量子电路模拟（通过 sampling/Train 间接依赖）
- **tqdm** — 进度条（可选）
