# Task 2 Mode 1 复现实施计划

> 仅指定模块与步骤，不实现代码。
>
> 目标：在 `n=4/6/8` 小规模上复现附录 D 的完整 Mode 1 流程：
> 构造 U_target(t) → 训练 local inversion → sewing → 验证 Z_j(t) 曲线。

---

## 已完成

| 文件 | 功能 |
|---|---|
| `config.py` | 参数集（n_qubits, seed, PhXZ/h_j 范围，时间点） |
| `U_target.py` | V_scr / H_diag / H_target / U_target(t) 构造与验证 |
| `circuit_plot.py` | Cirq 原生 SVG 电路图输出 |
| `circuit_plot_mpl.py` | Matplotlib 风格电路图输出（IDQNN 风格） |
| `U_target_reference.md` | 函数参考文档 |
| `cirq_usage_notes.md` | Cirq 使用约定文档 |

---

## 待实现

### 模块 A：Local Inversion Ansatz

**文件**：`task2_code/ansatz.py`

**功能**：
- 实现 4-qubit 参数化电路 `U_trial_Bj(theta)`（60 参数）
- 结构：`R(x,y,z) = cirq.rx(x) cirq.ry(y) cirq.rz(z)` + 固定 CZ 连接
- 门序：5 层单比特旋转 + 4 轮固定 CZ
- 接受 qubit 索引列表和 theta 向量，返回 `cirq.Circuit`

**依赖**：无（除 Cirq）

**验证**：
- 参数初始化在 `[0, 2π)` 随机
- 能通过 `cirq.unitary()` 提取 4-qubit 酉矩阵

---

### 模块 B：Superoperator / Partial Trace 工具

**文件**：`task2_code/superoperator.py`

**功能**：
- `vec(A)`：将矩阵 A 向量化（column-stacking）
- `superoperator(U)`：将酉 U 转为 column-stacking superoperator 表示 `U* ⊗ U`
- `partial_trace(rho_total, keep_indices, dims)`：对多 qubit 密度矩阵做偏迹
- `partial_trace_superoperator(S_total, keep_indices, dims)`：对 Liouville superoperator 按物理 qubit 做偏迹，输出可直接与 `I_4` 比较

**依赖**：NumPy

**验证**：
- `U = I` 时 superoperator = identity
- 单 qubit partial trace 与手算一致
- CZ 纠缠态偏迹后 reduced state 应为最大混合态

---

### 模块 C：Light Cone 提取

**文件**：`task2_code/lightcone.py`

**功能**：
- `extract_lightcone(H_target, block_Bj, radius)`：给定光锥半径，返回 `S_j` 内的子酉 `U_target_Sj`
- 利用 `H_target` 的局域结构做截断
- 对 n≤8 可直接从稠密 `U_target` 矩阵中 trace out 非光锥比特

**依赖**：`U_target.py`, `superoperator.py`

**验证**：
- `radius=0` 时 `S_j == B_j`
- 光锥外 qubit 初始化为 `|0⟩` 时结果等价于直接对全系统做偏迹

---

### 模块 D：Mode 1 Deterministic Loss

**文件**：`task2_code/local_loss.py`

**功能**：
- `compute_loss(U_target_Sj, U_trial_Bj, block_qubits)`：
  - 将 `U_trial_Bj` 嵌入光锥：`U_trial_tilde = U_trial_Bj ⊗ I_{Sj\Bj}`
  - 计算 `superoperator(U_target_Sj · U_trial_tilde^†)`
  - 对 block 内每个 qubit 做 `partial_trace_superoperator`
  - 与 `I_4`（identity superoperator）做 Frobenius 距离
  - 对 block 内所有 qubit 求和

**依赖**：`superoperator.py`, `lightcone.py`

**验证**（来自复现指导 sanity checks）：
- `U_trial = identity` 且 `U_target = identity` → loss = 0
- `U_trial = U_target` → loss ≈ 0
- `U_target` 乘全局相位 → loss 不变

---

### 模块 E：训练循环

**文件**：`task2_code/train_local_inversion.py`

**功能**：
- 对单个 `B_j` 优化 `compute_loss`：
  - 多随机重启（num_restarts = 5–10）
  - 优化器：ADAM（lr ∈ {0.1, 0.3, 0.5}）或自写梯度
  - 每步记录 loss、梯度范数
  - 选最低 loss 的 theta 作为该 block 的最优 local inversion
- 支持批量处理所有 blocks

**依赖**：`ansatz.py`, `local_loss.py`, `U_target.py`

**验证**：
- 优化后 loss 比随机初始化显著降低
- 不同时间 t 下均能训练出低 loss

---

### 模块 F：Sewing 通道

**文件**：`task2_code/sewing.py`

**功能**：
- `block_swap(block_indices)`：对给定的 system qubit 集合构造 block-wise SWAP
- `sew_channel(V_Bj_list, block_list, n_qubits)`：
  - 对每个 `B_j` 构造 `W_j = (V_Bj ⊗ I)^† · S_Bj · (V_Bj ⊗ I)`
  - 全局 SWAP S
  - 返回 sewn 2n-qubit unitary
- `apply_sewn_channel(sewn_U, rho_input)`：
  - system + ancilla 初始化为 `|0^n⟩⟨0^n|`
  - 施加 sewn channel
  - trace out ancilla

**依赖**：`ansatz.py`（已训练的 V_Bj 参数）

**验证**：
- 仅 1 个 block 时，sewn channel ≈ 直接施加该 V_Bj 的结果
- 误差界与 paper Theorem 7 一致

---

### 模块 G：验证与绘图

**文件**：`task2_code/verify.py`

**功能**：
- `compute_Zj_expectation(state, qubit_index)`：期望值 ⟨Z_j⟩
- `compare_with_exact(U_target, sewn_channel, test_states, times)`：
  - 对每个 t 计算 learned channel 与 exact U_target 在测试态上的 Z_j 差值
  - 输出 MAE / RMSE
- `plot_Zj_vs_time(times, exact_Z, learned_Z)`：
  - 类似论文 Fig. 3 风格：横轴 t，纵轴 ⟨Z_j⟩
  - 同一图上画 exact（实线）和 learned（十字/点）

**依赖**：`sewing.py`, `U_target.py`, Matplotlib

**验证**：
- 测试态：随机乘积态 `|s_0⟩⊗...⊗|s_{n-1}⟩`
- 输出图可直接对比论文 Fig. 3 的 qubit 3 / qubit 16 曲线

---

## 实施顺序与依赖图

```
已经完成
  │
  ├─ config.py
  └─ U_target.py
       │
       ▼
  ┌─ ansatz.py ────────────────────────────────┐
  │                                              │
  ├─ superoperator.py ──┐                        │
  │                      │                        │
  ├─ lightcone.py ──────┤                        │
  │                      ▼                        │
  │               local_loss.py ◄─────────────────┘
  │                      │
  │                      ▼
  │          train_local_inversion.py
  │                      │
  │                      ▼
  │               sewing.py
  │                      │
  │                      ▼
  └────────────── verify.py
```

建议按 A → B → C → D → E → F → G 顺序实现，每步完成后跑验证再进下一步。

---

## 里程碑

| 阶段 | 成功标准 |
|---|---|
| A+B | 参数化 ansatz → 酉矩阵，superoperator 工具通过 sanity check |
| C+D | n=4 下单 block 的 local loss 能计算且通过三个 sanity check |
| E | n=4 下单 block 训练后 loss < 1e-3 |
| F | n=4 下一个 block 的 sewn channel 能正确还原目标 qubit |
| G | 输出 n=4 的 Z_j(t) 对比图，短时长时间均可匹配 |
| 扩展 | n=6 或 n=8 的完整 pipeline 运行，验证多 block sewing |

---

## 暂不实现（超出第一版范围）

- sampled Pauli coefficient loss（Appendix F）
- 大 n 光锥截断（>12 qubit，需 state-vector 替代方案）
- 硬件噪声 / error mitigation
- TFQ 集成
- 全 20-qubit 稠密模拟
