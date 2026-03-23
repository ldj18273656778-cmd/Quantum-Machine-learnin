# 复现代码说明文档（可继续修改）

本文档用于帮助你快速理解并继续修改 `arXiv:2509.09033` 的 bitstring 生成复现代码。

## 1. 目录结构

- `idqnn_bitstring.py`
  - 核心实现文件（浅层模型 + 深层映射模型 + 评估工具函数）
- `reproduce_bitstring.py`
  - 复现实验入口脚本（固定 seed 运行并打印结果）
- `test_idqnn_correctness.py`
  - 正确性验证脚本（自动测试）
- `README.md`
  - 快速运行说明

## 2. 代码主线（先看这个）

最重要的两个采样函数：

1. `sample_shallow_idqnn(...)`
   - 对应论文 Appendix C.3.a 的 Algorithm 1（2D shallow）
   - 直接构建电路并采样

2. `sample_deep_mapped_idqnn(...)`
   - 对应论文 Appendix C.3.a 的 Algorithm 2（(1+1)D deep mapping）
   - 逐层推进量子态，生成输出 bitstring

建议阅读顺序：

1. `IDQNNConfig` / `make_config`：看参数含义
2. `build_shallow_circuit`：看浅层电路怎么搭
3. `sample_deep_mapped_idqnn`：看深层映射怎么生成每一位
4. `reproduce_bitstring.py`：看如何调用和复现

## 3. 参数含义与数据格式

- `n1`：时间维（深度维）长度
- `m`：每个时间切片上的 qubit 数
- 总位数 `n = n1 * m`
- `x`（输入 bitstring）采用行优先展平：
  - `x[t, q] -> x[t * m + q]`
- `theta` 形状必须是 `(n1, m)`，即每个 `(t, q)` 一个旋转角

## 4. 你最可能会改的地方

### 4.1 改图结构（非链式连接）

默认空间边是线性链 `(0-1-2-...)`。  
你可以改成任意自定义边：

```python
cfg = make_config(
    n1=4,
    m=5,
    spatial_edges=[(0, 2), (1, 3), (3, 4)],  # 自定义
    include_temporal_edges=True,
)
```

### 4.2 改输入分布 / 参数分布

在 `reproduce_bitstring.py` 里，`x` 和 `theta` 都是随机生成的。  
你可以直接替换成你想测试的固定实例。

### 4.3 增加评估指标

`idqnn_bitstring.py` 里已经有：
- `exact_probs_shallow`
- `linear_xeb`
- `empirical_distribution`
- `tv_distance`

你可以在此基础上加 KL、JS、局部统计一致性等指标。

## 5. 正确性验证怎么做

运行测试脚本：

```powershell
& "C:\ProgramData\anaconda3\python.exe" "code\bitstring_generation_2509_09033\test_idqnn_correctness.py"
```

测试会验证：

- 输入/输出维度是否正确
- 输出是否只包含 0/1
- 同 seed 是否完全可复现
- 不同 seed 是否会变化
- 浅层采样分布是否接近精确概率
- 一个固定参数下的回归指纹是否一致（防止改代码后悄悄漂移）

## 6. 修改后建议流程

每次你改完代码后，建议固定执行两步：

1. 跑测试：`test_idqnn_correctness.py`
2. 跑复现：`reproduce_bitstring.py`（看 hash、样本头部、位均值）

如果测试通过但复现指标明显变化，通常说明：
- 你改了算法行为（可能是有意的）
- 或者随机种子/输入生成逻辑被改动了

## 7. 当前环境注意事项

在这台机器上，`C:\ProgramData\anaconda3\envs\dwave\python.exe` 导入 `cirq` 会崩溃。  
建议统一使用：

`C:\ProgramData\anaconda3\python.exe`

