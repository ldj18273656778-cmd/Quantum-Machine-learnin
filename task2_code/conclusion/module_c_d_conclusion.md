# 模块 C/D 实现总结

## 范围

本次更新实现了 Mode 1 的首个小规模路径，包括：

- 模块 C：光锥提取工具，位于 `task2_code/lightcone.py`
- 模块 D：确定性局部反演损失函数，位于 `task2_code/local_loss.py`
- 纯 Python 验证脚本，位于 `task2_code/test_module_c_d.py`

训练循环、优化器、缝合（sewing）、绘图、大规模稀疏模拟以及完整的 reduced-channel 形式化均有意排除在外。

## 模块 C 设计

模块 C 将三个容易混淆的概念分开：

1. 光锥索引选取。
2. 稠密投影算子提取。
3. 检验投影算子是否表现为酉矩阵的诊断信息。

关键函数：

- `lightcone_qubits_for_block(block_qubits, n_qubits, radius=1)` 返回围绕目标块的、按升序排列且作边界裁剪的 1D 半径扩展光锥。
- `backward_lightcone_from_circuit(circuit, output_qubits)` 以逆时间顺序遍历 Cirq 操作，提供基于电路的因果光锥诊断。
- `projected_operator_on_lightcone(U_full, lightcone_qubits, n_qubits, outside_state=0, require_unitary=False, atol=1e-8)` 在选定量子比特上计算 `K_S = <0_out| U_full |0_out>`。
- `extract_target_lightcone_operator(...)` 返回一个 `LightConeResult`，其中包含投影算子、`lightcone_qubits`、`block_positions`、`outside_qubits`、`semantics` 以及诊断信息。

返回的算子使用 `semantics="outside_zero_projection"`。它不是对酉矩阵做 Hilbert 空间的偏迹，也不保证是真正的子系统酉矩阵。诊断信息记录：

- `||K^dag K - I||`
- `||K K^dag - I||`
- 投影子空间的最大列范数泄漏

稠密提取路径默认限制在 `n_qubits <= 8` 且 Hilbert 维度 `<= 256`，因此不会意外地将 `config.n_qubits = 20` 分配为稠密矩阵。

projected_operator_diagnostics 是对已经得到的矩阵做诊断，不做投影；
projected_operator_on_lightcone 是对完整系统矩阵 U_full 做 light cone 投影；
如果输入是 Cirq 电路，则由 extract_target_lightcone_operator 先把电路转换成矩阵，再调用 projected_operator_on_lightcone

## 模块 D 设计

模块 D 实现了实施计划中描述的酉算子损失：

```text
V = U_target_Sj @ U_trial_tilde^dag
S = superoperator(V)
loss = sum_r || partial_trace_superoperator(S, [r], [2]*|S_j|) - I_4 ||_F
```

关键函数：

- `embed_block_unitary_in_lightcone(U_block, lightcone_qubits, block_qubits)` 通过显式的基态索引操作，将块酉矩阵嵌入到按 `lightcone_qubits` 排序的 Hilbert 空间中。
- `compute_loss(U_target_Sj, U_trial_Bj, block_qubits, lightcone_qubits=None, require_unitary=True, atol=1e-8)` 计算 Mode 1 确定性损失。

实现使用了 `task2_code.superoperator.superoperator`，保留了仓库约定：

```text
superoperator(U) = U.conj() kron U
```

同时使用 `partial_trace_superoperator(..., normalize=True)`，因此较大光锥上的恒等通道会约化为单量子比特恒等超算子 `I_4`。

## 张量顺序约定

所有稠密矩阵使用升序量子比特顺序。基矢整数的解释与仓库中 `reduce(np.kron, ops)` 风格一致（大端序）：序号较小的量子比特位于张量积列表的前端，充当更显著的基矢位。

`block_qubits` 是全局标签（原始链上的绝对位置）。`block_positions` 是其在 `lightcone_qubits` 内的局部位置。模块 D 在嵌入或偏迹之前始终将标签转换为位置。

## 已运行的测试

命令：

```bash
python task2_code/test_code/test_module_c_d.py
```

结果：

```text
PASS test_lightcone_radius_zero_and_boundary
PASS test_project_identity_on_lightcone
PASS test_projection_detects_cross_boundary_nonunitarity
PASS test_embed_nonleading_block_position
PASS test_identity_and_global_phase_loss_zero
PASS test_exact_closed_support_loss_zero
PASS test_nonleading_exact_embedded_loss_zero
PASS test_invalid_inputs_raise
```

测试覆盖：

- 半径零和边界裁剪光锥
- 恒等矩阵投影
- 跨边界 CNOT 的非酉 outside-`|0>` 投影检测
- 非前导位置的块嵌入
- 恒等损失为零
- 闭合支持内精确 target/trial 损失为零
- 全局相位不变性
- 非法输入处理与稠密规模保护

## 局限性与后续工作

- 投影算子路径是小规模复现路径，不是通用的 open-system reduced channel。
- 数学上更一般的 reduced-channel 实现应返回目标超算子/通道而非 `U_target_Sj`，且需要修订模块 D 的通道组合接口。
- 基于电路的局部酉提取仅当选定光锥在电路操作下闭合时才严格成立。当前实现提供了电路因果光锥诊断，但尚未构建任意的闭合子电路。
- 模块 E 的训练、ADAM 优化、缝合及验证绘图仍为独立的未来模块。
