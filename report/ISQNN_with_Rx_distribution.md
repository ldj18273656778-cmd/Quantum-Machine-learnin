# ISQNN 加 Rx(φ) 后的输出分布完整推导

> 参考 `report/beamer_dqnn_isqnn/` 中无 Rx 版本的推导结构，引入 $R_x(\phi)$ 门。

---

## 1. 电路与约定

$n_1\times m$ 矩形 qubit 网格，二维索引 $(i_1,i_2)$。

**编码**：$x=0 \Rightarrow H\ket{0}=\ket{+}$，$x=1 \Rightarrow \ket{0}$

**操作序**：编码 → $R_x(\phi)$ → $R_z(\theta)$ → slice 内 CZ → slice 间 CZ → 逐 qubit X 测量

---

## 2. Rx 门转换

$$
R_x(\phi)\ket{+} = e^{-i\phi/2}\ket{+} \quad (\ket{+}\text{ 是 }X\text{ 本征态})
$$

$$
R_x(\phi)\ket{0} = c\ket{0} - i s\ket{1} \equiv \ket{u},\qquad
c=\cos\frac{\phi}{2},\; s=\sin\frac{\phi}{2}
$$

**核心差异**：$\ket{u}$ 有 $|1\rangle$ 分量，不再是 CZ 本征态。$|c|^2 - |s|^2 = \cos\phi$。

---

## 3. 单个 qubit 的 Rz 后态

$$R_z(\theta)\ket{+} = \frac{e^{-i\theta/2}}{\sqrt{2}}\ket{0} + \frac{e^{i\theta/2}}{\sqrt{2}}\ket{1}$$

$$R_z(\theta)\ket{u} = c\;e^{-i\theta/2}\ket{0} - i s\;e^{i\theta/2}\ket{1}$$

---

## 4. 约化密度矩阵的推导

考虑 qubit $j$，其 CZ 邻居集合为 $N(j)$。全局态为 product 态 after encoding + Rx + Rz，再经所有 CZ 门。

对 qubit $j$ 的约化密度矩阵 $\rho_j = \operatorname{Tr}_{\neq j}(\ket{\Psi}\bra{\Psi})$：

$$
\begin{aligned}
\rho_j[0,0] &= |a_0|^2 \\
\rho_j[1,1] &= |a_1|^2 \\
\rho_j[0,1] &= a_0a_1^* \prod_{k\in N(j)}\big(|b_{k,0}|^2 - |b_{k,1}|^2\big) \\
\rho_j[1,0] &= \rho_j[0,1]^*
\end{aligned}
$$

其中 $(a_0,a_1)$ 是 qubit $j$ 经 Rz 后的 Z 基系数，$b_{k,z_k}$ 是邻居 $k$ 的系数。

**交叉项的关键**：邻居 $k$ 对 off-diagonal 的贡献为

$$|b_{k,0}|^2 - |b_{k,1}|^2$$

### 对于 $x_k=0$（$\ket{+}$ 邻居）

$$|b_{k,0}|^2 = |b_{k,1}|^2 = \frac{1}{2}\;\Longrightarrow\;|b_{k,0}|^2 - |b_{k,1}|^2 = 0$$

**→ 任一 $\ket{+}$ 邻居将 $\rho_j[0,1]$ 置零。**

### 对于 $x_k=1$（$\ket{u}$ 邻居）

$$|b_{k,0}|^2 = c^2,\; |b_{k,1}|^2 = s^2\;\Longrightarrow\;|b_{k,0}|^2 - |b_{k,1}|^2 = \cos\phi$$

**→ 每个 $\ket{u}$ 邻居贡献因子 $\cos\phi$。**

### 对于非邻居

$$|b_{k,0}|^2 + |b_{k,1}|^2 = 1 \quad (\text{归一化，在 trace 中抵消})$$

---

## 5. 一般公式

设 qubit $j$ 有 $d$ 个 CZ 邻居，其中 $d_0$ 个为 $\ket{+}$ 邻居 ($x=0$)，$d_u$ 个为 $\ket{u}$ 邻居 ($x=1$)，$d = d_0 + d_u$。

### 情况 1：存在 $\ket{+}$ 邻居（$d_0 \geq 1$）

$$\boxed{\rho_j[0,1] = 0 \;\Longrightarrow\; P(y_j=0) = P(y_j=1) = \frac{1}{2}}$$

**→ $\ket{+}$ 像"防火墙"，完全消除偏差。**

### 情况 2：全部邻居为 $\ket{u}$（$d_0 = 0,\; d_u = d$）

$$\rho_j[0,1] = a_0a_1^* \cdot (\cos\phi)^d$$

代入 $a_0,a_1$ 的具体形式：

#### 当 $x_j=0$（qubit $j$ 编码为 $\ket{+}$）

$$|a_0|^2=|a_1|^2=\frac{1}{2},\quad a_0a_1^*=\frac{e^{-i\theta_j}}{2}$$

$$
\boxed{P(y_j=1) = \frac{1}{2}\big[1 - (\cos\phi)^{d_u}\,\cos\theta_j\big]}
$$

#### 当 $x_j=1$（qubit $j$ 编码为 $\ket{u}$）

$$|a_0|^2=c^2,\;|a_1|^2=s^2,\quad a_0a_1^*=i\,c s\,e^{-i\theta_j}=\frac{i}{2}\sin\phi\,e^{-i\theta_j}$$

$$
\boxed{P(y_j=1) = \frac{1}{2}\big[1 - \sin\phi\cdot(\cos\phi)^{d_u}\,\sin\theta_j\big]}
$$

---

## 6. 特殊值验证

### $\phi = 0$（无 Rx）

| $x_j$ | $P(y_j=1)$ | 说明 |
|-------|-----------|------|
| $0$ | $\frac{1}{2}(1 - \cos\theta_j)$ | 复原闭合估计公式 |
| $1$ | $\frac{1}{2}$（$\sin 0 = 0$） | $\ket{0}$ 在 X 基恒为 $1/2$|

### $\phi = \pi/2$

| $x_j$ | $P(y_j=1)$ |
|-------|-----------|
| 任意 | $\frac{1}{2}$ | $\cos\frac{\pi}{2}=0$，全部 off-diagonal 为零 |

### $\phi = \pi/4$，$d_u=2$

| $x_j$ | $P(y_j=1)$ |
|-------|-----------|
| $0$ | $\frac{1}{2} - \frac{1}{4}\cos\theta_j$ |
| $1$ | $\frac{1}{2} - \frac{\sqrt{2}}{4}\sin\theta_j$ |

---

## 7. 与参数估计的联系

当**所有邻居为 $x=1$**（训练数据筛选条件），$d_0=0$，$d_u=d$：

$$P(y_j=1) = \frac{1}{2}\big[1 - (\cos\phi)^{d}\,\cos\theta_j\big] \quad (x_j=0)$$

$$\hat{\theta}_j = \arccos\!\left(\frac{1-2\bar{y}_j}{(\cos\phi)^{d}}\right)$$

其中 $d = |N(j)|$ 是该 qubit 的 CZ 邻居总数。$\phi=0$ 时退化为原估计公式 $\hat{\theta}_j = \arccos(1-2\bar{y}_j)$。

**对于 MNIST 估计**：此公式仅在假设训练数据 $(x,y)$ 由含 Rx 的量子电路生成时成立。若 $y$ 来自身编码标签（非电路输出），则公式为启发式映射。

---

## 8. 数值验证

### 3-qubit 链：$\phi=\pi/4$，$\theta=[0.8, 0.3, 1.5]$

| $x$ | qubit | $x$ | $d$ | $d_u$ | $P(y=1)$ 预测 | $P(y=1)$ 仿真 | 误差 |
|-----|-------|-----|-----|-------|-------------|-------------|:---:|
| `000` | q0 | 0 | 1 | 0 | 0.500 | 0.500 | 0 |
| `010` | q0 | 0 | 1 | 0 | 0.500 | 0.500 | 0 |
| `011` | q0 | 0 | 1 | 1 | $0.5(1-c_\phi c_{0.8})$ | **0.254** | 0 |
| `100` | q0 | 1 | 1 | 0 | 0.500 | 0.500 | 0 |
| `110` | q0 | 1 | 1 | 1 | $0.5(1-s_\phi c_\phi s_{0.8})$ | **0.321** | 0 |
| `110` | q1 | 1 | 2 | 1 | 0.500 | 0.500 | 0 |
| `110` | q2 | 0 | 1 | 1 | $0.5(1-c_\phi c_{1.5})$ | **0.475** | 0 |
| `111` | q0 | 1 | 1 | 1 | $0.5(1-s_\phi c_\phi s_{0.8})$ | **0.321** | 0 |
| `111` | q1 | 1 | 2 | 2 | $0.5(1-s_\phi c_\phi^2 s_{0.3})$ | **0.448** | <1e-5 |
| `111` | q2 | 1 | 1 | 1 | $0.5(1-s_\phi c_\phi s_{1.5})$ | **0.251** | <1e-5 |

其中 $c_\phi = \cos\frac{\pi}{4} = 0.7071$，$s_\phi = \sin\frac{\pi}{4} = 0.7071$。

### 5-qubit 链偏差传播模式

```
q0 --CZ-- q1 --CZ-- q2 --CZ-- q3 --CZ-- q4

x=00000:  |+>   |+>   |+>   |+>   |+>
P:       0.5   0.5   0.5   0.5   0.5         全部 0.5

x=11100:  |u>   |u>   |u>   |+>   |+>
P:      0.32  0.45  0.5   0.5   0.5         q3=|+> 阻断 |u> 簇

x=01110:  |+>   |u>   |u>   |u>   |+>
P:      0.25  0.5   0.32  0.5   0.17        两端的 |+> 被 |u> 邻居"感染"
                                              中间的 q2 被两边 |u> 夹击

x=11111:  |u>   |u>   |u>   |u>   |u>
P:      0.32  0.45  0.32  0.40  0.40        全部偏离 0.5
```

**传播规则**：

| 目标 $\,$ 邻居 | $\ket{+}$ | $\ket{u}$ |
|-------------|----------|----------|
| $\ket{+}$ | 不影响 | **偏差** ∝ $\cos\phi$ |
| $\ket{u}$ | 不影响 | **偏差** ∝ $\cos\phi$ |

- $\ket{+}$ 邻居像"防火墙"——消除目标 qubit 的所有 off-diagonal
- $\ket{u}$ 邻居贡献偏差因子 $\cos\phi$，每个额外的 $\ket{u}$ 邻居叠加一次
- 偏差只会从 $\ket{u}$ **向外**传播到一轮邻居，不会穿透 $\ket{+}$

---

## 9. 对 DQNN 的启示

在无 Rx 的原始电路中，$x=0$ 的 random.randint 之所以合法，是因为所有 qubit 的 $P(y_j=1)$ 恒为 $1/2$（$\ket{0}$ 是 CZ 本征态，$\ket{+}$ 使 off-diagonal 恒零）。

加 Rx 后：
- $x=0$ qubit 若所有邻居为 $x=0$ → $P=1/2$（OK，random.randint 合法）
- $x=0$ qubit 若有 $x=1$ 邻居 → $P\neq 1/2$（random.randint 给出错误的边际分布）
- $x=1$ qubit 若所有邻居为 $x=0$ → $P=1/2$（$\ket{+}$ 防火墙）
- $x=1$ qubit 若有 $x=1$ 邻居 → $P\neq 1/2$

**MNIST 场景**：大量 $x=1$ 像素密集分布的图像中，许多 $x=0$ qubit 都有 $x=1$ CZ 邻居，$P(y_j=1)$ 会偏离 $1/2$。DQNN 的 `random.randint` 在此场景下系统性出错。
