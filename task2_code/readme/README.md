# task2_code — Task 2 Mode 1 复现代码库

> 当前推荐的运行方式、备份说明和兼容性约束见
> [`USAGE.md`](USAGE.md)。本目录保持原有脚本路径和 `task2_code.*`
> import 兼容；历史 `data/`、`module_e_results*` 和验证输出不会被自动迁移。

> 对应论文 *Generative quantum advantage for classical and quantum problems* Appendix D，
> Task 2 "Learning to generate compressed simulation circuits"。
>
> 所有函数均使用 Cirq 构建量子电路，PhXZ 使用 Cirq 原生 `cirq.PhasedXZGate`。

---

## 1. 基础矩阵常量

| 常量 | 形状 | 说明 |
|---|---|---|
| `I2` | (2, 2) | 单比特单位矩阵 |
| `Z` | (2, 2) | Pauli-Z 矩阵 |

---

## 2. PhasedXZ 门

使用 **Cirq 原生 `cirq.PhasedXZGate`**。

### 参数约定

```
cirq.PhasedXZGate(x_exponent=x, z_exponent=z, axis_phase_exponent=a)

等效电路：Z^{-a} X^x Z^a Z^z  （时间顺序）

酉矩阵：
  [ e^{iπx/2} cos(πx/2),           -i e^{iπ(x/2-a)} sin(πx/2)      ]
  [-i e^{iπ(x/2+z+a)} sin(πx/2),    e^{iπ(x/2+z)} cos(πx/2)       ]
```

参数为 **π 分数**（exponent 约定）：`x_exponent=1` 对应旋转 `π` rad（完整 π 旋转）。

### 与论文参数范围的关系

论文参数 `x, z, a ∈ [0, 4]` 为弧度角。转换为 Cirq exponent：

\[
x_{\mathrm{cirq}} = \frac{x_{\mathrm{paper}}}{\pi},\qquad \text{范围 } [0,\,4/\pi] \approx [0,\,1.273]
\]

### `random_phxz(rng)`

**输入**：`rng` — `np.random.Generator`

**输出**：`cirq.PhasedXZGate`，参数各自独立采样自 `[0, 4/π]`。

**作用**：生成一个随机 PhXZ 门，其物理旋转角范围与论文一致（≈ 0–1.27π rad）。

---

## 3. V_scr — 随机乱序酉

### `build_v_scr_circuit(n, rng)`

**输入**：`n` — qubit 数，`rng` — 随机数生成器

**输出**：`cirq.Circuit`

**作用**：按 Eq. S.2.6 构建 V_scr 电路，每层为单个 `cirq.PhasedXZGate`（1 moment）。

```
PhXZ(全) → CZ(even) → PhXZ(全) → CZ(odd) → PhXZ(全)
```

电路深度：**5**（3 层 PhXZ + 2 层 CZ，每层 PhXZ 是 1 个 moment）。

**优点**：使用原生 Cirq Gate，`cirq.inverse()` 自动支持，无需自定义类。

### `build_v_scr_unitary(n, rng)`

**输出**：`np.ndarray`，(2^n, 2^n)，V_scr 的稠密酉矩阵。

---

## 4. 哈密顿量

### `build_h_diag(n, rng)`

\[
H_{\mathrm{diag}} = \sum_{j} h_j Z_j,\qquad h_j \sim \mathrm{Uniform}[-1, 1]
\]

返回 `(H_diag, h_arr)`。

### `build_h_target(V_scr, H_diag)`

\[
H_{\mathrm{target}} = V_{\mathrm{scr}}^\dagger \; H_{\mathrm{diag}} \; V_{\mathrm{scr}}
\]

---

## 5. U_target(t) — 目标时间演化

### `build_u_target_circuit(n, rng, h_arr, t)`

电路时间顺序（论文 "U, Rz, U†"）：

```
V_scr → R_Z 层 → V_scr^†
```

V_scr^† 由 `cirq.inverse()` 自动生成。

R_Z 层：`cirq.rz(+2 h_j t)` 对应 `exp(-i h_j Z_j t)`。

### `u_target_unitary(n, rng, h_arr, t)`

提取稠密酉矩阵（调用 `cirq.unitary()`）。

### `u_target_expm(H_target, t)`

`scipy.linalg.expm(-i H_target t)`，用于基准验证。

---

## 6. 辅助函数

### `is_unitary(U, tol=1e-10)`

检查 `||U U^dag - I|| < tol`。

---

## 7. 4-qubit Local Inversion Ansatz (`ansatz.py`)

Eq. S.2.7 的参数化电路，用于学习局部反演。共 **60 个可训练参数**。

### 电路结构

```
Layer 1:  R00  R01  R02  R03          (Rx Ry Rz ×4)
          CZ(0,1)     CZ(2,3)
Layer 2:  R10  R11  R12  R13
               CZ(1,2)
Layer 3:  R20  R21  R22  R23
               CZ(1,2)
Layer 4:  R30  R31  R32  R33
          CZ(0,1)     CZ(2,3)
Layer 5:  R40  R41  R42  R43
```

### 函数

| 函数 | 说明 |
|---|---|
| `theta_count()` | 返回 60 |
| `random_theta(rng)` | 随机初始化 [0, 2π) |
| `build_ansatz_4q(theta, qubits)` | 返回 `cirq.Circuit` |
| `ansatz_unitary(theta, qubits)` | 返回 (16, 16) 酉矩阵 |

### 验证

| 检查 | 结果 |
|---|---|
| `U(0) == I` | `0.00e+00` |
| `U U^dag == I` | `3.53e-15` |
| 参数数目 | `60` |

---

## 8. Superoperator / Partial Trace 工具计划 (`superoperator.py`)

模块 B 只负责矩阵层工具，不构造 Cirq 电路。它连接 `ansatz.py`、后续 `lightcone.py` 和 `local_loss.py`：

```text
ansatz_unitary(theta) ─┐
                       ├─► local_loss.py ─► superoperator.py
U_target_Sj ───────────┘
```

### 8.1 数学约定

使用 column-stacking 向量化：

\[
\mathrm{vec}(A) = [A_{0,0}, A_{1,0}, \dots, A_{d-1,0}, A_{0,1}, \dots]^T
\]

也就是先按列展开矩阵。若密度矩阵演化为

\[
\rho' = U \rho U^\dagger,
\]

则向量化后有

\[
\mathrm{vec}(\rho') = (U^* \otimes U)\,\mathrm{vec}(\rho).
\]

因此 `superoperator(U)` 应返回 `U.conj() ⊗ U`，形状从 `(d, d)` 变为 `(d^2, d^2)`。

Partial trace 用于把多 qubit 总密度矩阵或 superoperator 张量缩减到指定子系统。它的核心作用是：

\[
\rho_{\mathrm{keep}} = \mathrm{Tr}_{\mathrm{discard}}(\rho_{\mathrm{total}}).
\]

在 Mode 1 loss 中，它会把光锥 `S_j` 上的 superoperator 缩减到 block 内单个 qubit 的 reduced superoperator，用来和 `I_4` 比较。

### 8.2 函数计划

| 函数 | 输入 | 输出 | 用途 |
|---|---|---|---|
| `vec(A)` | `A`: `np.ndarray`, shape `(d, d)` | `v`: shape `(d*d,)` | 将矩阵按 column-stacking 展平 |
| `unvec(v, dim)` | `v`: shape `(dim*dim,)`, `dim`: int | `A`: shape `(dim, dim)` | `vec` 的逆操作，用于调试和验证 |
| `superoperator(U)` | `U`: shape `(d, d)` 酉矩阵 | `S`: shape `(d*d, d*d)` | 构造 unitary channel 的矩阵表示 `U* ⊗ U` |
| `partial_trace(rho_total, keep_indices, dims)` | `rho_total`: shape `(D, D)`, `keep_indices`: 子系统编号, `dims`: 每个子系统维度 | `rho_keep`: shape `(D_keep, D_keep)` | 对普通矩阵/密度矩阵求偏迹 |
| `partial_trace_superoperator(S_total, keep_indices, dims, normalize=True)` | `S_total`: shape `(D^2, D^2)`, `dims`: 物理子系统维度 | reduced superoperator | 对 Liouville superoperator 按物理 qubit 求偏迹 |

### 8.3 输入输出细节

`vec(A)`：

- 输入必须是二维方阵 `(d, d)`。
- 输出是一维向量 `(d*d,)`。
- 展开顺序固定为 column-major，用于匹配 `U* ⊗ U` 公式。

`superoperator(U)`：

- 输入来自 `U_target_Sj` 或 `ansatz_unitary(theta)` 这类稠密酉矩阵。
- 若 `U` 是 4-qubit ansatz，输入形状为 `(16, 16)`，输出形状为 `(256, 256)`。
- 若 `U` 是光锥 `S_j` 上的目标酉，`d = 2^{|S_j|}`，输出形状为 `(d^2, d^2)`。

`partial_trace(rho_total, keep_indices, dims)`：

- `rho_total` 是总系统矩阵，形状必须为 `(prod(dims), prod(dims))`。
- `dims` 对 qubit 系统通常是 `[2, 2, ..., 2]`。
- `keep_indices` 使用与 `dims` 一致的子系统顺序，例如 `[0]` 表示保留第 0 个 qubit。
- 输出维度为 `prod(dims[i] for i in keep_indices)` 的平方矩阵。

`partial_trace_superoperator(S_total, keep_indices, dims, normalize=True)`：

- `S_total` 是 `superoperator(U)` 的输出，形状为 `(D^2, D^2)`。
- `dims` 仍然使用物理 Hilbert 空间维度，例如 3 个 qubit 用 `[2, 2, 2]`，不要手动传 `[4, 4, 4]`。
- 函数内部会把 column-stacking 的 Liouville 轴重排成每个物理 qubit 一个 4 维 operator space，再调用普通 `partial_trace`。
- 默认 `normalize=True`，因此 `partial_trace_superoperator(I, [q], [2]*n)` 返回 `I_4`，正好可用于 loss 中和 identity superoperator 比较。

### 8.4 与 loss function 的对接

`local_loss.py` 中的目标接口计划为：

```text
compute_loss(U_target_Sj, U_trial_Bj, block_qubits)
```

其中：

| 参数 | 来源 | 形状 | 含义 |
|---|---|---|---|
| `U_target_Sj` | `lightcone.py` | `(2^{|S_j|}, 2^{|S_j|})` | 光锥上的目标时间演化 |
| `U_trial_Bj` | `ansatz_unitary(theta)` | `(16, 16)` | 4-qubit local inversion ansatz |
| `block_qubits` | 训练循环 | 长度 4 的 qubit 索引 | 当前优化的 block |

loss 内部会使用模块 B：

```text
1. 将 U_trial_Bj 嵌入到 S_j：U_trial_tilde = U_trial_Bj ⊗ I
2. 计算误差酉：M = U_target_Sj · U_trial_tilde^dag
3. 转换为 superoperator：S = superoperator(M)
4. 对 block 内每个 qubit 做 partial_trace_superoperator
5. 与单 qubit identity superoperator I_4 做 Frobenius 距离
```

### 8.5 验证计划

| 检查 | 期望结果 |
|---|---|
| `unvec(vec(A), d)` | 返回原矩阵 `A` |
| `superoperator(I_d)` | 返回 `I_{d^2}` |
| `superoperator(e^{iφ} U)` | 与 `superoperator(U)` 完全相同 |
| 单 qubit Bell 态偏迹 | reduced state 为 `I/2` |
| `partial_trace(rho, keep_indices=all)` | 返回原矩阵 |
| `partial_trace(rho, keep_indices=[])` | 返回 trace 标量 `Tr(rho)` |
| `partial_trace_superoperator(I, [q], [2]*n)` | 返回单 qubit identity superoperator `I_4` |

### 8.6 实施边界

- 该模块只依赖 NumPy。
- 不引入 Cirq 对象，输入输出全部是 `np.ndarray`。
- 不处理训练、优化或光锥提取逻辑。
- 不在这里保存文件或绘图。
- qubit 顺序必须在 README、`lightcone.py`、`local_loss.py` 中保持一致，否则 partial trace 会得到错误子系统。

---

## 9. 快速调用示例

```python
import numpy as np
from task2_code.config import seed
from task2_code.U_target import (
    build_v_scr_unitary, build_h_diag, build_h_target,
    u_target_unitary, u_target_expm,
)

rng = np.random.default_rng(seed)
V = build_v_scr_unitary(n, rng)
Hd, h_vec = build_h_diag(n, rng)
Ht = build_h_target(V, Hd)

t = 0.5
rng = np.random.default_rng(seed)
U_cirq = u_target_unitary(n, rng, h_vec, t)
U_ex = u_target_expm(Ht, t)
assert np.allclose(U_cirq, U_ex)
```

---

## 10. 论文复现全流程

### 10.1 第一步：构造目标物理系统

```python
rng = np.random.default_rng(seed)
V = build_v_scr_unitary(n_qubits, rng)
Hd, h_vec = build_h_diag(n_qubits, rng)
Ht = build_h_target(V, Hd)
```

### 10.2 第二步：生成各时间的 U_target(t)

```python
times = [3 * math.pi / 40 * k + 0.001 for k in range(0, 41)]
for t in times:
    rng = np.random.default_rng(seed)
    U_t = u_target_unitary(n_qubits, rng, h_vec, t)
```

**关键**：每次必须用相同的 seed，确保不同 `t` 下 V_scr 一致。

### 10.3 第三步：block 划分与 local inversion loss（后续模块）

```python
blocks = [[0,1,2,3], [4,5,6,7], [8,9,10,11], [12,13,14,15], [16,17,18,19]]
for B_j in blocks:
    S_j = range(max(0, B_j[0]-3), min(20, B_j[-1]+4))
    # 提取 U_target_Sj，优化 4-qubit ansatz
```

### 10.4 第四步：Sewing 缝合

```python
# Round 1: V_B1, V_B3, V_B5
# Round 2: V_B2, V_B4
```

### 10.5 验证检查清单

| 检查项 | 方法 |
|---|---|
| V_scr 酉性 | `is_unitary(V)` |
| H_diag/H_target 同谱 | `eigvalsh(Hd) == eigvalsh(Ht)` |
| U_cirq == U_expm | `allclose` at ~1e-15 |
| U_target 酉性 | `is_unitary(U_t)` |

---

## 11. 调用关系图

```
config.py (参数)
  │
  ▼
U_target.py
  │
  ├─► random_phxz(rng)                   ── cirq.PhasedXZGate
  │
  ├─► build_v_scr_circuit(n, rng)         ── V_scr 电路
  ├─► build_v_scr_unitary(n, rng)         ── V_scr 稠密酉矩阵
  │
  ├─► build_h_diag(n, rng)                ── H_diag 矩阵 + h_j
  ├─► build_h_target(V_scr, H_diag)       ── H_target 矩阵
  │
  ├─► build_u_target_circuit(...)         ── U_target(t) Cirq 电路
  ├─► u_target_unitary(...)               ── U_target(t) 稠密酉
  ├─► u_target_expm(H_target, t)          ── 基准对比
  │
  └─► is_unitary(U)                       ── 酉性检查
```

---

## 12. 光锥测试

脚本 `lightcone_test.py` 提供两种方向、两种实现方法的光锥分析。

### 12.1 函数清单

| 函数 | 方向 | 方法 | n 限制 |
|---|---|---|---|
| `backward_lightcone_dense(U, q, n)` | 输出 q ← 输入 | 交换子（稠密矩阵） | n ≤ 10 |
| `forward_lightcone_dense(U, j, n)` | 输入 j → 输出 | 交换子（稠密矩阵） | n ≤ 10 |
| `backward_lightcone_circuit(circ, q, n)` | 输出 q ← 输入 | 电路反向追踪 | 任意 n |
| `forward_lightcone_circuit(circ, j, n)` | 输入 j → 输出 | 电路正向追踪 | 任意 n |

### 12.2 稠密交换子方法

**原理**：Heisenberg 绘景。

对后向光锥：`O_q = U^† · Z_q · U`，若 `||[O_q, Z_j]|| > 0`，则输入 j 影响输出 q。

对前向光锥：`F_j = U · Z_j · U^†`，若 `||[F_j, Z_q]|| > 0`，则输入 j 传播到输出 q。

**缺点**：需构造 2^n × 2^n 稠密酉矩阵，n ≥ 12 内存不可行。

### 12.3 电路追踪方法

**原理**：不依赖矩阵，直接在 Cirq 电路上逐 gate 追踪。

核心函数 `_gate_adds_to_lightcone(op, current_qubits)`：

```text
1. 提取 gate 的所有 qubit：qs = op.qubits
2. 判断 qs 与当前受影响的 qubit 集合 current 是否有交集
3. 有交集 → 信息通过此 gate 扩散 → current ∪ qs
4. 无交集 → gate 未触及光锥 → current 不变
```

**后向追踪** `backward_lightcone_circuit(circ, q, n)`：

```python
affected = {q}
for op in reversed(list(circuit.all_operations())):   # 从后往前
    qs = op.qubits
    if qs & affected:
        affected |= qs
return sorted(affected)
```

`reversed(list(circuit.all_operations()))` 将电路操作按**时间逆序**排列：输出端的最后一个 gate 最先被处理，追溯信息从输出 qubit 反向传播到输入端。

**前向追踪** `forward_lightcone_circuit(circ, j, n)`：

```python
affected = {j}
for op in circuit.all_operations():     # 从前往后
    qs = op.qubits
    if qs & affected:
        affected |= qs
return sorted(affected)
```

直接按电路时间正序遍历——输入 qubit 的信息沿 gate 正向扩散到输出端。

**优点**：不依赖稠密矩阵，n=20 甚至更大均可运行。已验证与稠密交换子方法完全一致。

**直观例子**（反向追踪 qubit 4）：

```text
初始: affected = {4}
CZ(3,4) → {3}∩{4}? 否，{3,4}∩{4}≠∅ → affected = {3,4}
CZ(2,3) → {2,3}∩{3,4}={3}≠∅    → affected = {2,3,4}
CZ(1,2) → {1,2}∩{2,3,4}={2}≠∅  → affected = {1,2,3,4}
PhXZ(5) → {5}∩{1,2,3,4}=∅       → 不变

返回: [1,2,3,4]  ← qubit 4 的信息来自输入 qubit 1-4
```

### 12.4 测试结果

**n=10 — 稠密与电路方法对比**（全部一致）：

```
         backward light cone          forward light cone
   q     dense         circuit        j     dense         circuit
   0     [0,1,2,3]     [0,1,2,3]  OK  0     [0,1,2,3]     [0,1,2,3]  OK
   4   [2,3,4,5,6,7] [2,3,4,5,6,7] OK  4   [2,3,4,5,6,7] [2,3,4,5,6,7] OK
   9     [6,7,8,9]     [6,7,8,9]  OK  9     [6,7,8,9]     [6,7,8,9]  OK
```

**n=20 — 仅电路方法**：

```
  backward q= 0: [0,1,2,3]           size=4
  backward q= 9: [6,7,8,9,10,11]    size=6
  backward q=19: [16,17,18,19]       size=4
```

**结论**：

- V_scr（深度 5）光锥半径 ≈ ±3 qubit，与论文 "light cone 3 qubits up/down" 一致
- 所有时间点光锥相同——R_z 层对角，纠缠仅源于 V_scr 的 CZ
- 中间 qubit 光锥最大（6 qubits），边缘 qubit 被边界截断（4 qubits）

**运行**：

```bash
python task2_code/test_code/lightcone_test.py
```
