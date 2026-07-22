# Cirq 使用细节

> `task2_code/U_target.py` 中涉及 Cirq 的关键约定与实现说明。

---

## 1. 已使用的 Cirq 原生门

| Cirq 门 | 矩阵定义 | 用途 |
|---|---|---|
| `cirq.rz(θ)` | `exp(-i Z θ/2)`，`θ` 单位 rad | PhXZ 的 Z 旋转、R_Z 演化层 |
| `cirq.rx(θ)` | `exp(-i X θ/2)`，`θ` 单位 rad | PhXZ 的 X 旋转 |
| `cirq.CZ(q0, q1)` | `diag(1, 1, 1, -1)`，自逆 | V_scr 的两比特纠缠层 |

---

## 2. PhXZ 门实现

### 论文定义

\[
\text{PhXZ} = Z^{-a} X^x Z^a Z^z \qquad (\text{时间顺序，参数 }x,z,a\in[0,4])
\]

### 转换为 Cirq 原生门

| 论文项 | 矩阵 | Cirq 等价 |
|---|---|---|
| `Z^{-a}` | `exp(+i a Z)` | `cirq.rz(-2a)` |
| `X^x` | `exp(-i x X)` | `cirq.rx(2x)` |
| `Z^a Z^z` | `exp(-i(a+z) Z)` | `cirq.rz(2(z+a))` |

**推导**：

- `cirq.rz(θ) = exp(-i Z θ/2)`，令 `θ = -2a` → `exp(+i a Z)` ✓
- `cirq.rx(θ) = exp(-i X θ/2)`，令 `θ = 2x` → `exp(-i x X)` ✓
- `cirq.rz(θ) = exp(-i Z θ/2)`，令 `θ = 2(z+a)` → `exp(-i(z+a) Z)` ✓

### 代码

```python
def phxz_gate(x, z, a):
    q = cirq.LineQubit(0)
    return cirq.Circuit([
        cirq.rz(-2 * a).on(q),          # Z^{-a}
        cirq.rx(2 * x).on(q),           # X^x
        cirq.rz(2 * (z + a)).on(q),     # Z^a Z^z
    ])
```

3 个 moment，每个 moment 一个原生门，因此 `cirq.inverse()` 自动支持取逆。

---

## 3. cirq.PhasedXZGate 与本文实现的关系

Cirq 自带 `cirq.PhasedXZGate(x_exponent, z_exponent, axis_phase_exponent)`。

| 属性 | cirq.PhasedXZGate | 本文实现 |
|---|---|---|
| 算式等价 | `Z^{-a} X^x Z^a Z^z` | 相同 |
| 参数语义 | **π 分数**（exponent=1 → 旋转 π rad） | **弧度**（值直接是角弧度） |
| 酉矩阵 | `e^{iπx/2} cos(πx/2)` 等 | `cos(x) I - i sin(x) X` 等 |
| `inv_gate` | 内置 | 因用原生 `rz/rx`，自动通过 `cirq.inverse()` |

**数值验证**：两个实现的矩阵通过简单线性缩放 (`/π` 等) 无法对齐——因为整体相位约定和三角函数参数化路径不同。本文选择用 `cirq.rz`/`cirq.rx` 直接实现，规避参数映射的不确定性，且已验证到机器精度。

---

## 4. V_scr 电路

```
PhXZ(全) → CZ(even) → PhXZ(全) → CZ(odd) → PhXZ(全)
```

每层 PhXZ 消耗 3 个 Cirq moment（rz → rx → rz），总计 3×3 + 2 = **11 个 Cirq moment**。在硬件上通过并行调度可压缩到深度 5。

CZ 在对偶 pair 间无重叠，可直接放入同一 moment：

| Moment | 内容 |
|---|---|
| 0-2 | `B1` PhXZ 旋转层（所有 qubit） |
| 3 | CZ even: (0,1), (2,3), ... |
| 4-6 | `B2` PhXZ 旋转层 |
| 7 | CZ odd: (1,2), (3,4), ... |
| 8-10 | `B3` PhXZ 旋转层 |

---

## 5. V_scr^† 的构造

由于全部使用 Cirq 原生门（`rz`, `rx`, `CZ`），逆电路通过以下方式自动生成：

```python
ops_dag = [cirq.inverse(op) for op in reversed(list(circ_v.all_operations()))]
circ_vdag = cirq.Circuit(ops_dag)
```

`cirq.inverse(cirq.rz(θ))` → `cirq.rz(-θ)`
`cirq.inverse(cirq.rx(θ))` → `cirq.rx(-θ)`
`cirq.inverse(cirq.CZ(...))` → `cirq.CZ(...)`（自逆）

---

## 6. U_target(t) 电路

```
V_scr → R_z 层 → V_scr^†
```

R_z 层操作：

\[
\exp(-i H_{\text{diag}} t) = \bigotimes_j \exp(-i h_j Z_j t)
\]

在 Cirq 中对应：

```python
cirq.rz(+2 * h_j * t)   # exp(-i h_j Z t) = R_Z(+2h t)
```

**推导**：`cirq.rz(θ) = exp(-i Z θ/2)`，令 `θ = 2h_j t` → `exp(-i h_j Z t)` ✓

---

## 7. `cirq.unitary()` 的 qubit 排序约定

Cirq 使用 **big-endian** 排序：`cirq.LineQubit(0)` 是最左侧的 tensor 因子。

即：
```
cirq.unitary(circ) ↔  Kron(q0_matrix, q1_matrix, ...)
```

这与本文 `build_h_diag` 中的 Kronecker 乘积顺序一致，已验证匹配。

---

## 8. 已知陷阱

| 陷阱 | 表现 | 解决 |
|---|---|---|
| `cirq.Circuit(list_ops)` 把所有操作放入**同一 moment** | 时间顺序丢失 | 使用 `circ += subcirc` 或 `circ.append(moment_ops)` |
| `cirq.inverse(circ)` 不能作用于 Circuit 对象 | `TypeError` | 改为对操作逐个取逆并反转顺序 |
| `cirq.PhasedXZGate` 参数语义与论文直读不兼容 | 矩阵不一致 | 使用 `cirq.rz`/`cirq.rx` 直接实现 |

---

## 9. 依赖

```text
cirq >= 1.0
numpy
scipy (用于 expm 基准验证)
```
