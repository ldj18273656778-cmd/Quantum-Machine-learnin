# Train 目录说明

本目录包含用于生成训练数据的脚本与输出文件。

## 脚本

- `generate_theta_demo.py`：生成示例参数文件 `theta_demo.npy`。
- `generate_xy_dataset.py`：使用固定的 `theta` 和分布 $D(x)$ 生成 $(x_i, y_i)^N$ 数据，并输出文本文件。
- `generate_Dx.py`：生成输入分布 $D(x)$ 的样本。

## 运行方式（手动改参数）

所有脚本都采用“脚本内固定参数”的方式，请直接在文件顶部的参数区修改：

- `N1`、`M`：模型规模
- `SEED`：随机种子
- `OUT_PATH` / `theta_path`：输出或输入路径

然后在项目根目录运行，例如：

- 生成 theta：运行 `generate_theta_demo.py`
- 生成 (x,y) 数据：运行 `generate_xy_dataset.py`

## 输出格式

- `xy_dataset.txt`：每行包含 `comp\tx\ty`，方便阅读。
- `theta_demo.npy`：形状为 `(N1, M)` 的浮点数组。

如需修改输出格式或路径，直接改脚本中的参数即可。
