# sampling

量子机器学习采样模块，实现 DQNN（Dense Quantum Neural Network）与 ISQNN（In-Slice Quantum Neural Network）两种量子电路生成及对比分析。

## 目录结构

```
sampling/
├── __init__.py                  # 包初始化，导出核心接口
├── __main__.py                  # `python -m sampling` 入口
├── main.py                      # 主演示脚本，同时调用 DQNN/ISQNN 生成器
├── cirq_circuit.py              # Cirq 量子电路构建工具函数
├── DQNN_generate_y.py           # DQNN y 值生成器
├── ISQNN_generate_y.py          # ISQNN y 值生成器 + 连通性描述
├── probility_distribution.py    # DQNN/ISQNN 输出概率分布采样与直方图
├── compare_effective_states.py  # DQNN vs ISQNN 逐层有效态密度矩阵对比
├── small_compare_example.py     # compare_effective_states 小规模使用示例
├── test_main.py                 # DQNN/ISQNN 生成器简洁测试
├── test_n1.py                   # ISQNN 多 n1 取值测试
├── test_prob.py                 # 概率分布生成 + TVD 指标 + 可视化测试
└── test1.ipynb                  # Jupyter 交互式测试笔记本
```

## 核心概念

- **bitstring x**：长度为 `n = n1 * m` 的二进制串，每位 `0` 表示该量子比特初始化为 $|+\rangle = H|0\rangle$，`1` 表示保持 $|0\rangle$
- **theta 参数**：长度 n 的旋转角列表，每个量子比特分配一个 Rz(θ) 旋转
- **DQNN**：逐层顺序处理，测量后重置量子比特（态坍缩 → 重置为 |0⟩），每层只有 m 个量子比特
- **ISQNN**：所有 n1×m 个量子比特同时存在，全并行测量，无重置操作，层间通过 CZ 门纠缠
- **CZ 连通模式**：偶数层连接 (0,1), (2,3)...；奇数层连接 (1,2), (3,4)...

---

## 各文件功能与可调用函数详解

### 1. `__init__.py` — 包初始化

导出核心接口 `DQNN_generate_y`、`ISQNN_generate_y`、`idqnn_connectivity`，并通过 `__getattr__` 惰性加载 `generate_probability_distribution`。

### 2. `cirq_circuit.py` — Cirq 量子电路构建工具

#### `create_initial_circuit(bitstring) -> (qubits, circuit)`

| 参数 | 类型 | 说明 |
|---|---|---|
| `bitstring` | `str` | 二进制串，`'0'` 位施加 H 门初始化为叠加态，`'1'` 位保持 |0⟩ |

| 返回值 | 类型 | 说明 |
|---|---|---|
| `qubits` | `list[cirq.LineQubit]` | 长度为 `len(bitstring)` 的量子比特列表 |
| `circuit` | `cirq.Circuit` | 仅含 H 门初始化的电路 |

#### `add_quantum_operations(circuit, qubits, theta, layers=1) -> circuit`

| 参数 | 类型 | 说明 |
|---|---|---|
| `circuit` | `cirq.Circuit` | 已有电路（原地修改） |
| `qubits` | `list[cirq.Qid]` | 量子比特列表 |
| `theta` | `float` 或 `list[float]` | Rz 旋转角度；标量则所有比特共用，列表则按序分配 |
| `layers` | `int` | 重复施加的层数，默认 1 |

| 返回值 | 类型 | 说明 |
|---|---|---|
| `circuit` | `cirq.Circuit` | 添加了 Rz 旋转和链式 CZ 门后的电路 |

#### `add_measurement(circuit, qubits) -> circuit`

| 参数 | 类型 | 说明 |
|---|---|---|
| `circuit` | `cirq.Circuit` | 已有电路（原地修改） |
| `qubits` | `list[cirq.Qid]` | 待测量的量子比特列表 |

| 返回值 | 类型 | 说明 |
|---|---|---|
| `circuit` | `cirq.Circuit` | 添加了 Z 基测量（key=`'result'`）的电路 |

#### `build_circuit(bitstring, theta, layers=1, measure=True) -> circuit`

| 参数 | 类型 | 说明 |
|---|---|---|
| `bitstring` | `str` | 二进制串 |
| `theta` | `float` 或 `list[float]` | Rz 旋转角度 |
| `layers` | `int` | 操作层数，默认 1 |
| `measure` | `bool` | 是否添加测量，默认 True |

| 返回值 | 类型 | 说明 |
|---|---|---|
| `circuit` | `cirq.Circuit` | 完整电路：初始化 + Rz + CZ + 可选测量 |

---

### 3. `DQNN_generate_y.py` — DQNN y 值生成器

#### `DQNN_generate_y(bitstring, n1, m, theta_list) -> (full_circuit, y)`

| 参数 | 类型 | 说明 |
|---|---|---|
| `bitstring` | `str` | 长度为 n1*m 的二进制串 |
| `n1` | `int` | 层数（block 数） |
| `m` | `int` | 每层量子比特数 |
| `theta_list` | `float` 或 `list[float]` | Rz 旋转参数列表，长度 n1*m |

| 返回值 | 类型 | 说明 |
|---|---|---|
| `full_circuit` | `cirq.Circuit` 或 `None` | 完整量子电路（cirq 不可用时返回 None） |
| `y` | `list[int]` | 生成的输出比特串，长度 n1*m |

**算法流程**：
1. 初始态 φ：对前 m 比特编码、Rz 旋转、CZ 纠缠（偶数对）
2. 逐块（block=1..n1-1）处理：若 x 位为 `0` 则随机采样并可能施加 X 门；若为 `1` 则测量 X 基后重置为 |0⟩
3. 每块处理完后施加该块的 Rz 旋转及偶/奇交替 CZ 纠缠
4. 最后一轮对 m 个比特全部测量 X 基

---

### 4. `ISQNN_generate_y.py` — ISQNN y 值生成器

#### `idqnn_connectivity(n1, m) -> G`

| 参数 | 类型 | 说明 |
|---|---|---|
| `n1` | `int` | 层数（slice 数） |
| `m` | `int` | 每层量子比特数 |

| 返回值 | 类型 | 说明 |
|---|---|---|
| `G` | `dict` | 连通结构字典，含键：`n1`、`m`、`n`（=n1*m）、`intra_slice_edges`（层内 CZ 边列表）、`inter_slice_edges`（层间 CZ 边列表）、`all_edges`（所有边） |

**说明**：bit 编号采用展平索引 `idx = slice_idx * m + i`。偶数层内部连接 `(2i, 2i+1)`，奇数层连接 `(2i+1, 2i+2)`；层间连接相邻 slice 同位置比特。

#### `ISQNN_generate_y(bitstring, n1, m, theta_list) -> (full_circuit, y)`

| 参数 | 类型 | 说明 |
|---|---|---|
| `bitstring` | `str` | 长度为 n1*m 的二进制串 |
| `n1` | `int` | 层数（slice 数） |
| `m` | `int` | 每层量子比特数 |
| `theta_list` | `float` 或 `list[float]` | Rz 旋转参数列表，长度 n1*m |

| 返回值 | 类型 | 说明 |
|---|---|---|
| `full_circuit` | `cirq.Circuit` 或 `None` | 完整量子电路（cirq 不可用时返回 None） |
| `y` | `list[int]` | 生成的输出比特串，长度 n1*m |

**算法流程**：
1. 将所有 n1*m 个量子比特排列为 n1 行 × m 列的 GridQubit 矩形网格
2. 全部比特按 bitstring 编码（`'0'` → H 门），施加各自的 Rz 旋转
3. 每层内部按偶/奇规则施加 CZ 门（与 DQNN 一致）
4. 相邻层间对应比特施加 CZ 纠缠
5. 获得整体态向量后，逐比特执行 X 基测量（H + Z 测量）并坍缩状态

---

### 5. `probility_distribution.py` — 概率分布采样与直方图

#### `bitlist_to_decimal(y_bits) -> int`

| 参数 | 类型 | 说明 |
|---|---|---|
| `y_bits` | `list[int]` | 二进制位列表（高位在前） |

| 返回值 | 类型 | 说明 |
|---|---|---|
| `decimal` | `int` | 转换后的十进制整数 |

#### `generate_probability_distribution(bitstring, n1, m, theta_list, num_samples=100, bin_width=1) -> (dqnn_samples, isqnn_samples, dqnn_hist, isqnn_hist, bins)`

| 参数 | 类型 | 说明 |
|---|---|---|
| `bitstring` | `str` | 输入二进制串，长度 n1*m |
| `n1` | `int` | 层数 |
| `m` | `int` | 每层量子比特数 |
| `theta_list` | `list[float]` | Rz 旋转参数列表 |
| `num_samples` | `int` | 对每种模型重复采样次数，默认 100 |
| `bin_width` | `int` | 直方图 bin 宽度，默认 1 |

| 返回值 | 类型 | 说明 |
|---|---|---|
| `dqnn_samples` | `list[int]` | DQNN 每次采样的 y 值（十进制） |
| `isqnn_samples` | `list[int]` | ISQNN 每次采样的 y 值（十进制） |
| `dqnn_hist` | `np.ndarray` | DQNN 直方图计数 |
| `isqnn_hist` | `np.ndarray` | ISQNN 直方图计数 |
| `bins` | `np.ndarray` | 直方图 bin 边界 |

---

### 6. `compare_effective_states.py` — DQNN vs ISQNN 逐层有效态密度矩阵对比

在密度矩阵形式下，对 DQNN 的逐层轨迹与 ISQNN 在相同测量历史下的约化密度矩阵进行定量对比（保真度、纯度、Frobenius 距离）。

#### 6.1 辅助数学函数

##### `rz_gate(theta) -> np.ndarray`

| 参数 | 类型 | 说明 |
|---|---|---|
| `theta` | `float` | 旋转角度（弧度） |

| 返回值 | 类型 | 说明 |
|---|---|---|
| `matrix` | `np.ndarray` | 形状 (2,2) 的 Rz(θ) 门矩阵，dtype complex128 |

##### `num_qubits_from_state(state) -> int`

| 参数 | 类型 | 说明 |
|---|---|---|
| `state` | `np.ndarray` | 态向量（或密度矩阵），元素数为 2 的幂 |

| 返回值 | 类型 | 说明 |
|---|---|---|
| `num_qubits` | `int` | 量子比特数 |

##### `normalize_state(state) -> np.ndarray`

| 参数 | 类型 | 说明 |
|---|---|---|
| `state` | `np.ndarray` | 态向量 |

| 返回值 | 类型 | 说明 |
|---|---|---|
| `normalized` | `np.ndarray` | 归一化后的态向量 |

##### `zero_state(num_qubits) -> np.ndarray`

| 参数 | 类型 | 说明 |
|---|---|---|
| `num_qubits` | `int` | 量子比特数 |

| 返回值 | 类型 | 说明 |
|---|---|---|
| `state` | `np.ndarray` | 全零态 |0...0⟩，形状 (2^num_qubits,) |

#### 6.2 单/双量子比特门操作

##### `apply_single_qubit_gate(state, gate, qubit, num_qubits=None) -> np.ndarray`

| 参数 | 类型 | 说明 |
|---|---|---|
| `state` | `np.ndarray` | 态向量 |
| `gate` | `np.ndarray` | 形状 (2,2) 的单比特门矩阵 |
| `qubit` | `int` | 目标比特索引（0-based） |
| `num_qubits` | `int` 或 `None` | 总比特数，None 则自动推断 |

| 返回值 | 类型 | 说明 |
|---|---|---|
| `new_state` | `np.ndarray` | 施加门操作后的态向量 |

##### `apply_two_qubit_gate(state, gate, qubit_a, qubit_b, num_qubits=None) -> np.ndarray`

| 参数 | 类型 | 说明 |
|---|---|---|
| `state` | `np.ndarray` | 态向量 |
| `gate` | `np.ndarray` | 形状 (4,4) 的双比特门矩阵 |
| `qubit_a` | `int` | 第一个目标比特索引 |
| `qubit_b` | `int` | 第二个目标比特索引（≠ qubit_a） |
| `num_qubits` | `int` 或 `None` | 总比特数 |

| 返回值 | 类型 | 说明 |
|---|---|---|
| `new_state` | `np.ndarray` | 施加门操作后的态向量 |

#### 6.3 测量与投影

##### `z_outcome_probabilities(state, qubit, num_qubits=None) -> (prob0, prob1)`

| 参数 | 类型 | 说明 |
|---|---|---|
| `state` | `np.ndarray` | 态向量 |
| `qubit` | `int` | 目标比特索引 |
| `num_qubits` | `int` 或 `None` | 总比特数 |

| 返回值 | 类型 | 说明 |
|---|---|---|
| `prob0` | `float` | 测量 Z 基得 0 的概率 |
| `prob1` | `float` | 测量 Z 基得 1 的概率 |

##### `project_z(state, qubit, outcome, num_qubits=None) -> (collapsed_state, probability)`

| 参数 | 类型 | 说明 |
|---|---|---|
| `state` | `np.ndarray` | 态向量 |
| `qubit` | `int` | 目标比特索引 |
| `outcome` | `int` | 投影结果，0 或 1 |
| `num_qubits` | `int` 或 `None` | 总比特数 |

| 返回值 | 类型 | 说明 |
|---|---|---|
| `collapsed_state` | `np.ndarray` | 投影并归一化后的态向量 |
| `probability` | `float` | 该结果的概率 |

##### `measure_x_with_forced_outcome(state, qubit, outcome, num_qubits=None) -> (collapsed_state, probability)`

| 参数 | 类型 | 说明 |
|---|---|---|
| `state` | `np.ndarray` | 态向量 |
| `qubit` | `int` | 目标比特索引 |
| `outcome` | `int` | 强制 X 基测量结果，0 或 1 |
| `num_qubits` | `int` 或 `None` | 总比特数 |

| 返回值 | 类型 | 说明 |
|---|---|---|
| `collapsed_state` | `np.ndarray` | 投影并归一化后的态向量 |
| `probability` | `float` | 该结果的概率 |

##### `reset_measured_qubit(state, qubit, outcome, num_qubits=None) -> np.ndarray`

| 参数 | 类型 | 说明 |
|---|---|---|
| `state` | `np.ndarray` | 态向量 |
| `qubit` | `int` | 目标比特索引 |
| `outcome` | `int` | 测量结果，0 或 1；1 时施加 X 门（相当于重置为 |0⟩） |
| `num_qubits` | `int` 或 `None` | 总比特数 |

| 返回值 | 类型 | 说明 |
|---|---|---|
| `new_state` | `np.ndarray` | 操作后的态向量 |

#### 6.4 密度矩阵工具

##### `reduced_density_matrix(state, keep_qubits, num_qubits=None) -> np.ndarray`

| 参数 | 类型 | 说明 |
|---|---|---|
| `state` | `np.ndarray` | 态向量 |
| `keep_qubits` | `list[int]` | 保留的比特索引列表 |
| `num_qubits` | `int` 或 `None` | 总比特数 |

| 返回值 | 类型 | 说明 |
|---|---|---|
| `rho` | `np.ndarray` | 约化密度矩阵，形状 (2^k, 2^k)，k = len(keep_qubits) |

##### `density_from_state(state) -> np.ndarray`

| 参数 | 类型 | 说明 |
|---|---|---|
| `state` | `np.ndarray` | 纯态向量 |

| 返回值 | 类型 | 说明 |
|---|---|---|
| `rho` | `np.ndarray` | 密度矩阵 ρ = |ψ⟩⟨ψ| |

##### `density_purity(rho) -> float`

| 参数 | 类型 | 说明 |
|---|---|---|
| `rho` | `np.ndarray` | 密度矩阵 |

| 返回值 | 类型 | 说明 |
|---|---|---|
| `purity` | `float` | Tr(ρ²)，范围 [0,1] |

##### `frobenius_distance_to_pure_state(psi, rho) -> float`

| 参数 | 类型 | 说明 |
|---|---|---|
| `psi` | `np.ndarray` | 纯态向量 |
| `rho` | `np.ndarray` | 密度矩阵 |

| 返回值 | 类型 | 说明 |
|---|---|---|
| `distance` | `float` | Frobenius 距离 ∥ |ψ⟩⟨ψ| − ρ ∥_F |

##### `pure_state_fidelity(psi, rho) -> float`

| 参数 | 类型 | 说明 |
|---|---|---|
| `psi` | `np.ndarray` | 纯态向量 |
| `rho` | `np.ndarray` | 密度矩阵 |

| 返回值 | 类型 | 说明 |
|---|---|---|
| `fidelity` | `float` | ⟨ψ|ρ|ψ⟩，范围 [0,1] |

#### 6.5 ISQNN 状态构建

##### `normalize_theta_list(theta_list, total_length) -> list[float]`

| 参数 | 类型 | 说明 |
|---|---|---|
| `theta_list` | `float` 或 `list[float]` | 参数（标量则广播） |
| `total_length` | `int` | 期望长度 |

| 返回值 | 类型 | 说明 |
|---|---|---|
| `values` | `list[float]` | 长度为 total_length 的浮点列表 |

##### `apply_slice_local_ops_to_state(state, theta_list, slice_index, m) -> np.ndarray`

| 参数 | 类型 | 说明 |
|---|---|---|
| `state` | `np.ndarray` | m 比特态向量 |
| `theta_list` | `list[float]` | 完整 theta 列表 |
| `slice_index` | `int` | 层索引 |
| `m` | `int` | 每层量子比特数 |

| 返回值 | 类型 | 说明 |
|---|---|---|
| `new_state` | `np.ndarray` | 施加该层 Rz + 偶数 CZ 后的归一化态向量 |

##### `slice_local_unitary(theta_list, slice_index, m) -> np.ndarray`

| 参数 | 类型 | 说明 |
|---|---|---|
| `theta_list` | `list[float]` | 完整 theta 列表 |
| `slice_index` | `int` | 层索引 |
| `m` | `int` | 每层量子比特数 |

| 返回值 | 类型 | 说明 |
|---|---|---|
| `U` | `np.ndarray` | 该层本地酉矩阵，形状 (2^m, 2^m) |

##### `prepare_slice_input_state(bitstring, slice_index, m) -> np.ndarray`

| 参数 | 类型 | 说明 |
|---|---|---|
| `bitstring` | `str` | 完整 bitstring |
| `slice_index` | `int` | 层索引 |
| `m` | `int` | 每层量子比特数 |

| 返回值 | 类型 | 说明 |
|---|---|---|
| `state` | `np.ndarray` | 仅该层比特按 bitstring 初始化的 m 比特态向量 |

##### `build_isqnn_pre_measurement_state(bitstring, n1, m, theta_list, omit_local_slice) -> np.ndarray`

| 参数 | 类型 | 说明 |
|---|---|---|
| `bitstring` | `str` | 完整 bitstring，长度 n1*m |
| `n1` | `int` | 层数 |
| `m` | `int` | 每层量子比特数 |
| `theta_list` | `list[float]` | 旋转参数 |
| `omit_local_slice` | `int` 或 `None` | 省略该层本地 Rz+CZ 操作（用于对比 pre_local） |

| 返回值 | 类型 | 说明 |
|---|---|---|
| `state` | `np.ndarray` | 测量前 n1*m 比特的全局态向量 |

##### `conditional_isqnn_slice_density(bitstring, n1, m, theta_list, current_slice, readout_history) -> (rho_pre_local, history_probability)`

| 参数 | 类型 | 说明 |
|---|---|---|
| `bitstring` | `str` | 完整 bitstring |
| `n1` | `int` | 层数 |
| `m` | `int` | 每层量子比特数 |
| `theta_list` | `list[float]` | 旋转参数 |
| `current_slice` | `int` | 当前层索引（≥1） |
| `readout_history` | `list[list[int]]` | 前 current_slice 层每层 m 个测量结果 |

| 返回值 | 类型 | 说明 |
|---|---|---|
| `rho_pre_local` | `np.ndarray` | 当前层的约化密度矩阵（pre_local） |
| `history_probability` | `float` | 该测量历史的联合概率 |

#### 6.6 DQNN 轨迹追踪与对比

##### `trace_dqnn_layers(bitstring, n1, m, theta_list, trajectory_seed) -> (state, layer_traces, final_x_readout)`

| 参数 | 类型 | 说明 |
|---|---|---|
| `bitstring` | `str` | 完整 bitstring |
| `n1` | `int` | 层数 |
| `m` | `int` | 每层量子比特数 |
| `theta_list` | `list[float]` | 旋转参数 |
| `trajectory_seed` | `int` | DQNN 随机轨迹的种子 |

| 返回值 | 类型 | 说明 |
|---|---|---|
| `state` | `np.ndarray` | 最终态向量 |
| `layer_traces` | `list[DQNNLayerTrace]` | 每层的轨迹记录（含 layer_index, readout, pre_local_state, post_local_state） |
| `final_x_readout` | `list[int]` | 最终 X 基测量结果（长度 m） |

##### `compare_layer_states(dqnn_trace, rho_pre_local, local_unitary, tol) -> dict`

| 参数 | 类型 | 说明 |
|---|---|---|
| `dqnn_trace` | `DQNNLayerTrace` | DQNN 某层的轨迹记录 |
| `rho_pre_local` | `np.ndarray` | ISQNN 当前层 pre_local 约化密度矩阵 |
| `local_unitary` | `np.ndarray` | 该层的本地酉矩阵（用于计算 rho_post_local） |
| `tol` | `float` | 判断状态"相同"的 Frobenius 距离容差 |

| 返回值 | 类型 | 说明 |
|---|---|---|
| `result` | `dict` | 含键：`pre_same`, `post_same`, `pre_fidelity`, `post_fidelity`, `pre_purity`, `post_purity`, `pre_distance`, `post_distance` |

##### `run_single_trial(bitstring, n1, m, theta_list, trajectory_seed, tol) -> dict`

| 参数 | 类型 | 说明 |
|---|---|---|
| `bitstring` | `str` | 完整 bitstring |
| `n1` | `int` | 层数 |
| `m` | `int` | 每层量子比特数 |
| `theta_list` | `list[float]` | 旋转参数 |
| `trajectory_seed` | `int` | 轨迹种子 |
| `tol` | `float` | 容差 |
| `reports` | `list[dict]` | 每次 trial 的完整报告，含 `final_x_readout` 和 `layers` 列表 |

#### 6.7 I/O 与辅助

##### `format_bits(bits) -> str`

| 参数 | 类型 | 说明 |
|---|---|---|
| `bits` | `list[int]` | 比特列表 |

| 返回值 | 类型 | 说明 |
|---|---|---|
| `s` | `str` | 拼接为字符串，如 `[0,1,1]` → `"011"` |

##### `parse_theta_argument(theta_json, total_length, seed) -> list[float]`

| 参数 | 类型 | 说明 |
|---|---|---|
| `theta_json` | `str` 或 `None` | JSON 标量或列表；None 则随机生成 |
| `total_length` | `int` | theta 期望长度 |
| `seed` | `int` | 随机种子 |

| 返回值 | 类型 | 说明 |
|---|---|---|
| `theta_list` | `list[float]` | 长度为 total_length 的 theta 列表 |

##### `validate_bitstring(bitstring, n1, m) -> None`

| 参数 | 类型 | 说明 |
|---|---|---|
| `bitstring` | `str` | 待验证的二进制串 |
| `n1` | `int` | 层数 |
| `m` | `int` | 每层比特数 |

验证 bitstring 长度等于 n1*m 且只含 `'0'` 和 `'1'`，否则抛出 `ValueError`。

##### `print_trial_report(trial_index, report) -> None`

打印一次 trial 的详细对比结果（各层的 readout、pre/post 保真度/纯度/距离）。

##### `summarize_reports(reports) -> None`

打印所有 trial 的汇总统计（总层数、pre/post 匹配数）。

##### `build_argument_parser() -> argparse.ArgumentParser`

构建命令行参数解析器，支持 `--bitstring`、`--n1`、`--m`、`--theta-json`、`--theta-seed`、`--trajectory-seed`、`--trials`、`--tol`。

---

### 7. `small_compare_example.py` — 小规模对比示例

`compare_effective_states` 的固定参数调用示例（n1=3, m=2, theta_list=[0.2, 0.6, 1.0, 1.4, 1.8, 2.2]），**无对外可调用函数**（纯脚本）。

---

### 8. 测试文件

| 文件 | 说明 |
|---|---|
| `test_main.py` | DQNN/ISQNN 生成器快速测试（n1=3, m=4），固定随机种子 42 |
| `test_n1.py` | 测试 `ISQNN_generate_y` 在不同 n1 取值（2/3/4）下的行为 |
| `test_prob.py` | 概率分布生成 + TVD 指标 + 直方图可视化，保存至 `output_images/` |
| `test1.ipynb` | Jupyter Notebook 交互式测试 |

---

## 使用方法

### 直接运行

```bash
python code/sampling/main.py
python code/sampling/test_prob.py
python code/sampling/small_compare_example.py
python code/sampling/compare_effective_states.py --bitstring 000100 --n1 3 --m 2 --trials 5
```

### 作为模块导入

```python
from sampling import DQNN_generate_y, ISQNN_generate_y, idqnn_connectivity
from sampling.probility_distribution import generate_probability_distribution

circuit, y = DQNN_generate_y("0011", n1=2, m=2, theta_list=[0.1, 0.2, 0.3, 0.4])
circuit, y = ISQNN_generate_y("0011", n1=2, m=2, theta_list=[0.1, 0.2, 0.3, 0.4])
G = idqnn_connectivity(n1=3, m=4)
```

## 依赖

- **cirq** — Google 量子计算框架（必需）
- **numpy** — 数值计算
- **matplotlib** — 图表绘制（test_prob.py）
- **tqdm** — 进度条（可选，probility_distribution.py 会降级处理）
