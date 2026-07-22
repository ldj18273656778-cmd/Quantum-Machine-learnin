# 模块 E：训练循环 — 思路与数据流

> 记录于 `conclusion/`，供审阅后转入实现。

---

## 1. 整体架构

```
                     config.py
                    (n, seed, time_list, h_range, lr, restarts, ...)
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
    U_target.py     lightcone.py     ansatz.py
    U_target(t)      K_S, S_j       U_trial(θ) 60参数
         │               │               │
         └───────┬───────┘               │
                 ▼                       ▼
         train_local_inversion.py
                 │
          For each t:
           For each block B_j:
            For restart 1..R:
              ADAM优化 θ → 最小化 loss
           保存 best θ_j(t)
                 │
                 ▼
          输出: θ*_j(t)  (best local inversion per block per time)
```

---

## 2. 数据流（单次梯度步）

```
θ (60维向量)
  │
  ├─► ansatz_unitary(θ) ─► U_trial(16×16)
  │
  ├─► embed_block_unitary_in_lightcone(U_trial, S_j, B_j) ─► U_trial_tilde (2^s × 2^s)
  │
  ├─► V = K_S @ U_trial_tilde^dag
  │
  ├─► 损失计算：
  │     ├─ s ≤ 6: compute_loss(V, B_j, S_j) → superoperator(V) = V* ⊗ V
  │     └─ s > 6: per_bit_losses_from_V(V, B_j, S_j) → 逐比特直接约化
  │
  └─► loss = Σ_{q∈B_j} ‖Tr_{≠q} S − I₄‖_F   → 标量
```

---

## 3. 损失函数的选择策略

| 光锥大小 s | 全超算子 (`superoperator.py`) | 逐比特直接约化 (`per_bit_losses_from_V`) | 训练选用 |
|---|---|---|---|
| s ≤ 6 | ✓ 可行 (≤128 MB) | ✓ 可行 | 全超算子（更快） |
| 7 ≤ s ≤ 10 | ✗ 不可行 | ✓ 可行 | 逐比特直接约化 |
| s > 10 | ✗ | ✓ 但慢 | 逐比特 |

> n=12、radius=3 → s=10，训练时必须走 `per_bit_losses_from_V`。

---

## 4. 梯度计算策略

ansatz 有 60 个参数，损失函数非凸。

| 方案 | 每步前向次数 | 精度 | 实现难度 |
|---|---|---|---|
| 有限差分 (central diff) | 120 (60×2) | ε=1e-4 时 ≈1e-8 | 极简 |
| 同时扰动随机近似 (SPSA) | 2 | 粗糙 | 简单 |
| 自动微分 (JAX/Autograd) | 1 | 精确 | 需引入新依赖 |

**建议先用有限差分**：

- 零额外依赖，与现有 NumPy/Cirq 栈兼容
- n=4 验证时单次 loss 极快（毫秒级），120 次评估 < 1 秒
- n=12 时 per-bit 单次 loss 约若干秒，120 次评估 ≈ 几分钟

```python
def numerical_gradient(theta, loss_fn, eps=1e-4):
    loss0 = loss_fn(theta)
    grad = np.zeros_like(theta)
    for k in range(len(theta)):
        theta[k] += eps
        loss_plus = loss_fn(theta)
        theta[k] -= 2 * eps
        loss_minus = loss_fn(theta)
        theta[k] += eps          # 恢复
        grad[k] = (loss_plus - loss_minus) / (2 * eps)
    return grad
```

---

## 5. ADAM 更新

纯 NumPy，不引入 PyTorch/TensorFlow：

```
输入: θ_init, loss_fn, lr, beta1=0.9, beta2=0.999, max_steps, gtol

m = 0向量, v = 0向量
for step = 1..max_steps:
    grad = numerical_gradient(θ, loss_fn)
    if ||grad|| < gtol:  break

    m = beta1 * m + (1-beta1) * grad
    v = beta2 * v + (1-beta2) * grad²
    m_hat = m / (1 - beta1^step)
    v_hat = v / (1 - beta2^step)
    θ -= lr * m_hat / (√v_hat + 1e-8)

    记录 loss, ||grad||

输出: θ*, loss_history
```

---

## 6. 多随机重启策略

```
For each block B_j at time t:
    best_theta = None, best_loss = inf

    For restart = 1..R:
        θ_init = random_theta(rng)         # [0, 2π)
        θ_opt, loss_history = ADAM(θ_init, loss_fn, ...)
        final_loss = loss_history[-1]

        if final_loss < best_loss:
            best_loss = final_loss
            best_theta = θ_opt

        if best_loss < 目标阈值:  early stop

    保存 best_theta_j(t)
```

---

## 7. n=12 训练的可行性分析

| 操作 | 耗时估算 |
|---|---|
| 1 次 `per_bit_losses_from_V` (s=10) | ~若干秒（4 qubit × 4 基底 × 512 env × 1024² matmul） |
| 1 次梯度 (60 参数中心差分) | 120 次 loss ≈ 几分钟 |
| 1 次 ADAM 迭代 | 同上 |
| 100 步 ADAM | ~数小时 |
| 1 个 restart | ~数小时 |

**建议分两步走**：

- **第一步（快速验证 n=4）**：n=4，block={0,1,2,3}，radius=0，s=4。超算子 256×256，单次 loss < 毫秒，100 步 ADAM < 10 秒。验证完整训练 pipeline。
- **第二步（扩展到 n=12）**：n=12，radius=3，s=10。必须走 `per_bit_losses_from_V`，但训练框架完全一致，仅替换 loss 函数。

---

## 8. `loss_fn` 闭包设计

```python
def make_loss_fn(K_S, block_qubits, lightcone_qubits, use_per_bit=True):
    """返回 loss_fn(theta)，其中 K_S 和 block/cone 已固定"""

    if use_per_bit:
        trial_tilde_builder = lambda U_trial: embed_block_unitary_in_lightcone(
            U_trial, lightcone_qubits_lst, block_qubits_lst
        )
        def loss_fn(theta):
            U_trial = ansatz_unitary(theta)
            trial_tilde = trial_tilde_builder(U_trial)
            V = K_S @ trial_tilde.conj().T
            bit_losses = per_bit_losses_from_V(V, block_qubits_lst, lightcone_qubits_lst)
            return sum(bit_losses.values())
    else:
        def loss_fn(theta):
            U_trial = ansatz_unitary(theta)
            return compute_loss(K_S, U_trial, block_qubits_lst, lightcone_qubits_lst)

    return loss_fn
```

这样梯度循环中每步只做 θ → loss 映射，其余开销（光锥提取、block/cone 预处理）只在闭包创建时发生一次。

---

## 9. 输出格式

训练完成后输出数据结构：

```python
results = {
    0.5: {
        (5,6,7,8): np.array([...60个参数...]),
        (1,2,3,4): np.array([...]),
    },
    1.0: { ... },
}
```

直接对接 Module F（sewing）：`sew_channel` 需要每个 block 在各时间点训练好的 `V_Bj = ansatz_unitary(theta)`。

---

## 10. 待确认事项

1. 梯度方案：有限差分 / SPSA / 自微分？
2. n=12 训练时间成本：每 restart 数小时是否可接受，或先跑 n=4 验证再扩展？
3. `per_bit_losses_from_V` 中 `col_idx = c*2 + d` 需修正为 `c + 2*d`（column-stacking 约定），是否先修复再训练？
4. 输出格式 `results[t][block]` 是否符合 Module F 的需求？
