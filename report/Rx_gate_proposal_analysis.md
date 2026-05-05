# 引入 Rx 门解决 IDQNN 全局信息传播问题的数学分析

> 基于用户提出的方案：在电路中加入 Rx 门打乱量子信息，使信息能够全局传播，
> 同时保留对 Rz 参数的闭合形式估计能力。

---

## 1. 问题回顾：|0⟩ 编码如何阻断信息传播

### 1.1 当前电路的编码规则

根据 `code/sampling/DQNN_generate_y.py` 和 `ISQNN_generate_y.py`：

- `x_j = 0` → 施加 H 门 → 编码为 `|+⟩ = (|0⟩+|1⟩)/√2`
- `x_j = 1` → 不施加门 → 保持 `|0⟩`

### 1.2 电路结构

```
Encode(x) → Rz(θ) → CZ(layer_1) → Rz(θ) → CZ(layer_2) → ... → X-measure → y
```

CZ 门作用在 IDQNN 连通图的边上：

```
分层内连接 (intra-slice)：
  - 偶数层：(0,1), (2,3), (4,5), ...   (m//2 对)
  - 奇数层：(1,2), (3,4), (5,6), ...   ((m-1)//2 对)

层间连接 (inter-slice)：
  - 相邻层对应位置：(slice_k[i], slice_{k+1}[i])
```

### 1.3 |0⟩ 是本征态的数学证明

CZ 门定义为：

$$CZ = |0⟩⟨0| \otimes I + |1⟩⟨1| \otimes Z$$

当控制量子比特为 |0⟩ 时：

$$CZ(|0⟩ \otimes |\psi\rangle) = |0⟩ \otimes |\psi\rangle \quad\text{(恒等)}$$

当目标量子比特为 |0⟩ 时（CZ 对称，也可交换角色）：

$$CZ(|\psi\rangle \otimes |0\rangle) = |\psi\rangle \otimes |0\rangle \quad\text{(恒等，因为 } Z|0\rangle = |0\rangle\text{)}$$

**结论**：CZ 门两端任一量子比特处于 |0⟩ 状态时，CZ 退化为恒等运算。

### 1.4 后果：连通分量独立

设输入比特串 x，其 0-1 表示对应量子态中的 |+⟩/|0⟩ 编码。所有被 |0⟩ 编码比特包围的 |+⟩ 块形成独立的 CZ-连通分量：

```
x:  [0 0 0] [1 1] [0 0] [1] [0 0 0 0]
    +------+      +----+     +--------+
    块A           块B         块C
```

- 块 A 内的所有比特通过 CZ 相关联 → 可以产生块内相关性
- 块 A 与块 B 之间被连续的 |0⟩ 比特隔离 → **统计独立**
- 不同块的 y 输出 → **factorize**（因子分解）

这便是 IDQNN 无法产生全局手写数字图像的根本原因：输出分布分解为独立子块的乘积，无法形成全局模式。

---

## 2. Rx 门方案的数学分析

### 2.1 提议电路

```
Encode(x) → Rx(φ) → Rz(θ) → CZ → X-measure → y
```

在原有电路的第 1 步（编码）和第 2 步（Rz 旋转）之间插入 Rx 门。

### 2.2 Rx 门的量子力学性质

Rx 门的矩阵表示：

$$R_x(\phi) = e^{-i\phi X/2} = \cos(\phi/2)I - i\sin(\phi/2)X = \begin{pmatrix} \cos(\phi/2) & -i\sin(\phi/2) \\ -i\sin(\phi/2) & \cos(\phi/2) \end{pmatrix}$$

#### 性质 2a：Rx 作用于 |0⟩

$$R_x(\phi)|0\rangle = \cos(\phi/2)|0\rangle - i\sin(\phi/2)|1\rangle$$

当 φ = π/2 时：

$$R_x(\pi/2)|0\rangle = \frac{|0\rangle - i|1\rangle}{\sqrt{2}} \quad\text{(等概率叠加态！)}$$

**关键**：|1⟩ 分量的出现意味着该比特可通过 CZ 向邻居施加 Z 运算，从而**打破阻断**。

#### 性质 2b：Rx 作用于 |+⟩

$$R_x(\phi)|+\rangle = e^{-i\phi/2}|+\rangle$$

证：∵ |+⟩ 是 X 的本征态（特征值 +1），Rx 是 X 轴旋转，仅添全局相位。

**推理后果**：X 基测量忽略该全局相位，因此 Rx 不影响 `x_j=0` 比特的输出期望值。

#### 性质 2c：Rx 与 X 测量

施加 Rx 后的 X 基测量：

$$\langle X \rangle' = \text{Tr}(X \cdot R_x(\phi)\rho R_x^\dagger(\phi)) = \text{Tr}(R_x^\dagger(\phi) X R_x(\phi) \cdot \rho)$$

由于 `[Rx(φ), X] = 0`（两者均关于 X 轴），有：

$$R_x^\dagger(\phi) X R_x(\phi) = X$$

=> X 期望值在 Rx 下**不变**。

**推论**：将 Rx 放在 Rz 和 CZ 之后、测量之前，不会改变输出分布。因此 Rx 必须放在 CZ 之前才能发挥作用。

#### 性质 2d：Rz 与 CZ 的对易性

Rz(θ) 和 CZ 均在 Z 基下是对角的：

$$R_z(\theta) = \begin{pmatrix} e^{-i\theta/2} & 0 \\ 0 & e^{i\theta/2} \end{pmatrix}, \quad CZ = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & -1 \end{pmatrix}$$

可以验证：$(R_z(\theta) \otimes I) \cdot CZ = CZ \cdot (R_z(\theta) \otimes I)$

类似地：$(I \otimes R_z(\theta)) \cdot CZ = CZ \cdot (I \otimes R_z(\theta))$

**推论**：Rz(θ) 可自由移动到 CZ 之前或之后，与 CZ 的位置交换不改变整体酉运算。因此下面两个电路等价：

```
Encode → Rx → Rz → CZ → Measure  ⇔  Encode → Rx → CZ → Rz → Measure
```

---

## 3. 闭合形式参数估计的可行性分析

估计目标的数学条件：需要找到一种输入配置，使得目标比特 j 的输出分布 $P(y_j)$ **仅依赖于 $\theta_j$**，不受其他参数影响。

### 3.1 邻居无 Rx（φ_neighbor = 0）——可行 ✓

**电路配置**：

- 目标比特 j：`x_j=0` → `|+⟩` → Rx(φ_j) → Rz(θ_j) → CZ → X测量
- 邻居比特 k：`x_k=1` → `|0⟩` → Rx(0)=不施加 → Rz(θ_k) → CZ → ...

**分析**：

邻居保持 |0⟩（φ_k = 0），CZ 退化为恒等运算：

$$CZ\big(R_z(\theta_j)|+\rangle_j \otimes |0\rangle_k\big) = R_z(\theta_j)|+\rangle_j \otimes |0\rangle_k$$

目标比特 j 的自由演化：

$$|\psi_j\rangle = R_z(\theta_j)R_x(\phi_j)|+\rangle = e^{-i\phi_j/2} R_z(\theta_j)|+\rangle = e^{-i\phi_j/2}\big(\cos(\theta_j/2)|+\rangle - i\sin(\theta_j/2)|-\rangle\big)$$

X 基测量概率：

$$P(y_j = 1) = |\langle-|\psi_j\rangle|^2 = \sin^2(\theta_j/2) = \frac{1 - \cos\theta_j}{2}$$

**估计公式（与原始一致）**：

$$\hat{\theta}_j = \arccos(1 - 2\bar{y}_j), \quad \bar{y}_j = \frac{1}{N_{sp}} \sum_i y_j^{(i)}$$

**数值验证**（对所有 θ ∈ [0, π] 均精确恢复）：

| θ_true | P(y=1) | θ_hat | 误差 |
|--------|--------|-------|------|
| 0.0000 | 0.0000 | 0.0000 | 0 |
| 0.7854 | 0.1464 | 0.7854 | 0 |
| 1.5708 | 0.5000 | 1.5708 | 0 |
| 2.3562 | 0.8536 | 2.3562 | 0 |
| 3.1416 | 1.0000 | 3.1416 | 0 |

### 3.2 邻居有 Rx（φ_neighbor ≠ 0）——结果取决于 φ 取值

**电路配置**：邻居也施加 Rx(φ_k)。

#### 3.2.1 严格推导

邻居态演化：

$$|\psi_k\rangle = R_z(\theta_k)R_x(\phi_k)|0\rangle = \cos(\phi_k/2)\cdot e^{-i\theta_k/2}|0\rangle - i\sin(\phi_k/2)\cdot e^{i\theta_k/2}|1\rangle$$

概率分布：
- 邻居处于 |0⟩ 的概率：$\cos^2(\phi_k/2)$
- 邻居处于 |1⟩ 的概率：$\sin^2(\phi_k/2)$

**|0⟩ 分量**（概率 cos²(φ_k/2)）：
- CZ 在邻居 = |0⟩ 时恒等，目标比特不受影响
- $P(y_j=1 \mid |0\rangle\text{-comp}) = \sin^2(\theta_j/2)$

**|1⟩ 分量**（概率 sin²(φ_k/2)）：
- CZ 向目标比特施加 Z：$Z \cdot R_z(\theta_j)|+\rangle$
- $Z|+\rangle = |-\rangle$，$Z|-\rangle = |+\rangle$
- $Z\big(\cos(\theta_j/2)|+\rangle - i\sin(\theta_j/2)|-\rangle\big) = \cos(\theta_j/2)|-\rangle - i\sin(\theta_j/2)|+\rangle$
- $P(y_j=1 \mid |1\rangle\text{-comp}) = |\langle-|\ldots|^2 = \cos^2(\theta_j/2)$

**综合概率**：

$$P(y_j=1) = \cos^2(\phi_k/2) \cdot \sin^2(\theta_j/2) + \sin^2(\phi_k/2) \cdot \cos^2(\theta_j/2)$$

化简为：

$$P(y_j=1) = \frac{1}{2} - \frac{1}{4}\big[\cos(\phi_k - \theta_j) + \cos(\phi_k + \theta_j)\big]$$

#### 3.2.2 特殊 φ 值下的行为

| φ_k | P(y_j=1) | θ 可辨识？ | 估计公式 |
|-----|----------|-----------|---------|
| 0 | $\frac{1-\cos\theta}{2}$ | ✓ | $\hat{\theta} = \arccos(1-2\bar{y})$ |
| π/4 | $\frac{1}{2} - \frac{\sqrt{2}}{4}\cos\theta$ | ✓ (需解 cosθ) | $\hat{\theta} = \arccos\big(\sqrt{2}(1-2\bar{y})\big)$ |
| **π/2** | **$\frac{1}{2}$** | **✗ 不可辨识！** | 输出恒为 50%，与 θ 无关 |
| 3π/4 | $\frac{1}{2} + \frac{\sqrt{2}}{4}\cos\theta$ | ✓ | 需变形公式 |
| π | $\frac{1+\cos\theta}{2}$ | ✓ | $\hat{\theta} = 2\arccos(\sqrt{\bar{y}})$ 或 $2\arccos(-\sqrt{\bar{y}})$ |

**关键发现**：φ_k = π/2 是最差情况——测量结果完全随机（50/50），θ 信息完全丢失。

#### 3.2.3 一般 φ 值的反解

对于 φ ∉ {0, π/2, π}，$P(y=1)$ 是 cos(θ) 和 sin(θ) 的线性组合：

$$P = \frac{1}{2} - \frac{\cos\phi_k}{2}\cos\theta + \frac{\sin\phi_k}{2}\sin\theta$$

可解出 θ，但公式较复杂。最简做法是**估计阶段不启用邻居的 Rx 门**（φ_neighbor = 0）。

### 3.3 完整的多邻居情况

在 IDQNN 中，每个比特 j 可能有多个邻居（邻接表来自 `idqnn_connectivity` 函数）。对于估计 θ_j，需要**所有邻居**均处于 CZ 本征态。考虑到 CZ 作用于邻居对 (j,k)，只要邻居 k 处于 |0⟩，该 CZ 门不影响 j 的测量结果。

当所有邻居的 Rx 门被关闭（φ_neighbor = 0）时，上述分析直接推广到多邻居情况，估计仍然有效。

---

## 4. 核心挑战：训练与推理的电路不匹配

### 4.1 问题的数学表述

根据上述分析，一个自洽的方案需要满足：

- **估计阶段**：φ_neighbor ∈ {0, π, 2π, …}（邻居保持计算基矢态，是 CZ 本征态）
- **推理阶段**：φ = π/2 等非平凡值（打破 |0⟩ 的阻断，实现全局信息传播）

两个阶段的量子线路不同：

$$\begin{aligned}
U_{\text{train}} &= U_{\text{CZ}} \cdot U_{\text{Rz}}(\theta) \cdot U_{\text{Rx}}(0)_{\text{neighbors}} \cdot U_{\text{enc}} \\
U_{\text{infer}} &= U_{\text{CZ}} \cdot U_{\text{Rz}}(\theta) \cdot U_{\text{Rx}}(\pi/2)_{\text{all}} \cdot U_{\text{enc}}
\end{aligned}$$

因此：

$$P_{\text{train}}(y \mid x_{\text{special}}) \neq P_{\text{infer}}(y \mid x_{\text{arbitrary}})$$

### 4.2 不匹配的可能后果

- **正面视角**：Rx 门提升了电路的表达能力。θ 参数虽然是在 φ=0 条件下估计的，但在 φ≠0 条件下仍然控制同样的 Rz 旋转参数。信息传播能力的提升（从局域 → 全局）可能远超分布偏移带来的损失。
- **负面视角**：φ 的不一致改变了有效哈密顿量。估计阶段学到的 θ 是针对无 Rx 电路的，在新电路下可能不再给出正确的条件分布 $P(y|x)$。

这是一个**经验问题**，需要实现后看数值结果。

---

## 5. 四种具体方案

### 方案 A：两阶段法（推荐优先尝试）

```
估计阶段: Encode → Rx(φ_j=π/2, φ_neighbors=0) → Rz(θ) → CZ → Measure
推理阶段: Encode → Rx(φ_all=π/2) → Rz(θ) → CZ → Measure
```

- 估计公式不变：$\hat{\theta}_j = \arccos(1 - 2\bar{y}_j)$
- φ_j（目标比特上的 Rx）可以是任意值（|+⟩ 是 Rx 本征态，不影响估计）
- 推理时所有比特的 Rx 均开启（φ = π/2），打破 |0⟩ 阻断
- **优点**：实现最简单，估计公式完全不变
- **风险**：存在电路不匹配

### 方案 B：统一 φ = π 方案

```
估计和推理均使用: Encode → Rx(φ=π) → Rz(θ) → CZ → Measure
```

- Rx(π)|0⟩ = -i|1⟩（仍是计算基矢态 → 仍是 CZ 本征态）
- 修改后的估计公式：$\hat{\theta} = 2\arccos(\sqrt{\bar{y}})$ 或 $2\arccos(-\sqrt{\bar{y}})$
- **无电路不匹配**
- **问题**：|1⟩ 同样是 Z 本征态（$Z|1\rangle = -|1\rangle$），CZ 使其邻居获得 -1 相位，但邻居的局部态（|+⟩ 或 |0⟩ 的概率振幅）不受影响。经过所有 CZ 后，单位特仍处于计算基矢态或 |+⟩ 态，阻断效应仍然存在！

**方案 B 不能解决全局传播问题**。

### 方案 C：Rx 放在 CZ 之后（无效方案）

```
Encode → Rz(θ) → CZ → Rx(φ) → Measure
```

- 估计不变（Rz 与 CZ 对易）
- 但 X 测量下 Rx 的作用被消除（性质 2c），推理也无增益
- **方案 C 无效**。

### 方案 D：多层交错 Rx-CZ 结构

```
Encode → Rx(φ₁) → Rz(θ₁) → CZ → Rx(φ₂) → Rz(θ₂) → CZ → ... → X-measure
```

- 类似通用变分量子线路
- 每层可有独立的 Rx 和 Rz 参数
- **问题**：多层 CZ 下闭合形式估计失效
- 需采用梯度下降等变分优化方法替代闭合估计
- 复杂度显著增加，失去原始方案轻量化的核心优势

---

## 6. 建议的实验方案

### 6.1 最小验证实验

1. 构造一个小规模 IDQNN（如 n1=3, m=4, n=12）
2. 用合成数据训练：生成 (x, y) 数据集，已知真实 θ 参数
3. 实现方案 A，对比两种推理模式：
   - **原模式**：φ = 0（使用 `DQNN_generate_y` 或 `ISQNN_generate_y`）
   - **新模式**：φ_all = π/2，在编码后、Rz 前插入 Rx 门
4. 评估指标：
   - y 输出分布的全局相关性（互信息、卡方距离等）
   - 不同 x 输入下 y 的条件分布是否更符合训练数据

### 6.2 MNIST 全规模实验

1. 在 10×10 MNIST 二值化数据集上运行方案 A
2. 关注指标：
   - 生成图像是否出现**全局性**特征（原本是完全碎片化的）
   - 与标签的匹配度（分类精度）
   - 不同 φ 值对生成质量的影响

### 6.3 Rx 参数作为超参数调优

- 固定 φ 候选值：{0, π/4, π/3, π/2, 2π/3, 3π/4}
- 估计阶段仍用 φ=0（保证闭合估计），推理阶段扫描不同 φ
- 寻找最佳的全局相关性与局部精度之间的平衡点

---

## 7. 总结与讨论

| 问题 | 结论 |
|------|------|
| Rx 是否能打破 |0⟩ 阻断？ | **是**。Rx(φ)|0⟩ 产生 |1⟩ 分量，使 CZ 能施加 Z 操作，信息得以传播 |
| 闭合估计能否维持？ | **取决于 φ**。φ_neighbor=0 时估计完全不变；φ_neighbor=π/2 时 θ 不可辨识；φ_neighbor=π 时可用修正公式 |
| 训练/推理不匹配是否致命？ | **不确定，需实验验证**。Rz(θ) 与 CZ 对易 + Rx 仅改变编码 + 表达能力提升可能弥补分布偏移 |
| 最优先尝试哪个方案？ | **方案 A**（两阶段法，φ_neighbor=0 估计 + φ_all=π/2 推理） |

### 开放问题

1. **是否存在某种 φ 值和编码方案的组合，使训练与推理电路完全一致？**
   例如：用不同的输入编码（非 |0⟩/|+⟩）配合固定的 φ = π/2，使得估计时邻居仍处于 CZ 本征态。

2. **是否需要放弃闭合估计，转向变分训练？**
   如果方案 A 的分布不匹配太严重，可能需要考虑方案 D——用梯度下降联合训练 Rx 和 Rz 参数。

3. **多层 Rx 的数学结构？**
   Rx(φ₁) → Rz(θ₁) → CZ → Rx(φ₂) → Rz(θ₂) → CZ 整体酉运算的解析性质值得进一步研究。

---

## 附录：当前代码中的关键函数

| 文件 | 函数 | 作用 |
|------|------|------|
| `sampling/DQNN_generate_y.py` | `DQNN_generate_y()` | 用 DQNN 线路生成 y（含动态测量） |
| `sampling/ISQNN_generate_y.py` | `ISQNN_generate_y()` | 用 ISQNN 线路生成 y（全量子） |
| `sampling/ISQNN_generate_y.py` | `idqnn_connectivity()` | 生成 CZ 连通图 G |
| `Train/find_x_indices_by_graph_condition.py` | `find_indices()` | 筛选满足 x_j=0 且邻居 x_k=1 的样本 |
| `Train/find_x_indices_by_graph_condition.py` | `build_adjacency()` | 从边列表构建邻接表 |
| `Train/estimate_theta_from_filtered_samples.py` | `estimate_theta_from_filtered_samples()` | 闭合估计单个 θ_j |
