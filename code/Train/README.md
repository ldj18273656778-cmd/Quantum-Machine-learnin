# Train 目录说明

本目录包含两类功能：
1) 训练数据生成；
2) 基于筛选样本的参数估计（按位估计 $\theta$）。

---

## 一、数据生成相关脚本

- `generate_theta_demo.py`：生成示例参数文件 `theta_demo.npy`。
- `generate_Dx.py`：生成输入分布 $D(x)$ 的样本。
- `generate_xy_dataset.py`：使用固定 $\theta$ 和分布 $D(x)$ 生成数据集 $(x_i, y_i)_{i=1}^N$。

主要输出：
- `data/theta_demo.npy`
- `data/xy_dataset.npy`
- `data/xy_dataset.txt`（便于人工查看）

---

## 二、参数估计相关脚本

### 1) `find_x_indices_by_graph_condition.py`

功能：按图局部条件筛选样本索引。

其中会先构建图的 `adjacency`（邻接表）：

- `adjacency[j]` 表示第 `j` 个比特的所有邻居索引集合；
- 由连边集合 `G["all_edges"]` 构造，若存在边 `(a,b)`，则将 `b` 加入 `adjacency[a]`，并将 `a` 加入 `adjacency[b]`。

筛选时使用 `neighbors = sorted(adjacency[target_bit])` 获得目标位邻居。

给定目标比特 `target_bit=j`，筛选满足：

$$
x_j=0,\quad x_{\mathcal N(j)}=1
$$

其中 $\mathcal N(j)$ 为图邻居集合。输出匹配样本的索引（0-based）。

主要输出：
- `data/matched_indices.txt`

### 2) `estimate_theta_from_filtered_samples.py`

功能：先调用上面的筛选逻辑，再对每个参数位进行估计，并循环得到全部参数矩阵。

单个参数估计公式：

$$
\hat\theta_j=\frac{1}{2}\arccos\!\left(1-\frac{2}{N_{\mathrm{sp}}}\sum_{t=1}^{N_{\mathrm{sp}}}y_t^{(j)}\right)
$$

其中 $N_{\mathrm{sp}}$ 是筛选后样本数，$y_t^{(j)}\in\{0,1\}$。

脚本会在主程序中遍历全部目标位 `j=0,...,n-1`，输出：
- `theta_hat_matrix_rad`（形状 `(n1, m)`）
- `theta_hat_matrix_deg`
- 每个位的统计信息（`N_sp`, `sum_y`, 邻居、索引等）

主要输出：
- `data/theta_estimate_all_bits.npy`

---

## 三、运行方式（手动改参数）

当前脚本采用“脚本内参数区”方式配置。请直接修改每个脚本 `if __name__ == "__main__":` 下的参数，例如：

- `n1`, `m`：模型规模
- `input_path`, `output_path`：输入/输出文件路径
- `target_bit_1based`：目标位（单参数调试时）

然后在项目根目录运行对应脚本即可。

---

## 四、数据文件说明

- `xy_dataset.npy`：通常是字典对象（`allow_pickle=True`），含键：`x`, `y`, `comps`, `theta`, `n1`, `m`, `seed`。
	- `x`：形状 `(N,)` 的比特串数组（字符串）
	- `y`：形状 `(N, n)` 的二值数组
- `theta_estimate_all_bits.npy`：参数估计输出字典，含估计矩阵与逐位统计结果。
