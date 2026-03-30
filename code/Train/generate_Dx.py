from __future__ import annotations

"""Generate input bitstrings x ~ D(x) for task-1 in arXiv:2509.09033.

Paper's D(x) (Appendix C.2.a):
1) with prob 1/3: x = 0^n
2) with prob 1/3: each bit i.i.d. P(0)=0.6, P(1)=0.4
3) with prob 1/3: each bit i.i.d. P(0)=0.2, P(1)=0.8
"""

import numpy as np
from pathlib import Path


def sample_dx(n_bits: int, num_samples: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (x_strings, component_ids).

    - x_strings: shape (num_samples,), each item is an n_bits-length bitstring.
    - component_ids: shape (num_samples,), values in {0,1,2}.
    """
    if n_bits <= 0: #
        raise ValueError("n_bits must be > 0")
    if num_samples <= 0:
        raise ValueError("num_samples must be > 0")

    rng = np.random.default_rng(seed)  # random number generator
    comps = rng.integers(0, 3, size=num_samples, dtype=np.int8) #numpy 默认int64，指定int8节省内存，足够存0/1/2三个类别；各1/3概率

    x = np.empty(num_samples, dtype=f"<U{n_bits}")#创建一个空的字符串数组，dtype指定为长度为n_bits的Unicode字符串
    all_zero = "0" * n_bits #预先生成全零字符串

    # Component 0: x = 0^n
    mask0 = comps == 0 #布尔掩码，标记哪些样本属于component 0
    if np.any(mask0):
        x[mask0] = all_zero #布尔索引，按条件批量操作”，取出 x 中所有 mask0=True 的元素，并将它们设置为全零字符串

    # Component 1: iid with P(0)=0.6
    mask1 = comps == 1
    n1 = int(np.sum(mask1)) #计算属于component 1的样本数量
    if n1 > 0:
        bits1 = rng.choice(["0", "1"], size=(n1, n_bits), p=[0.6, 0.4])#
        x[mask1] = np.array(["".join(row) for row in bits1], dtype=f"<U{n_bits}")#"".join得到python列表，array(..., dtype=...)将其转换为numpy字符串数组

    # Component 2: iid with P(0)=0.2
    mask2 = comps == 2
    n2 = int(np.sum(mask2))
    if n2 > 0:
        bits2 = rng.choice(["0", "1"], size=(n2, n_bits), p=[0.2, 0.8])
        x[mask2] = np.array(["".join(row) for row in bits2], dtype=f"<U{n_bits}")

    return x, comps


def generate_Dx(n: int, num_samples: int = 10000, seed: int = 7) -> np.ndarray:
    """Programmatic API for D(x):

    Usage:
        x = generate_Dx(n)

    Returns:
        `x` as a 1D numpy vector with shape `(num_samples,)`, where each
        entry is an n-bit string.
    """
    x, _ = sample_dx(n_bits=n, num_samples=num_samples, seed=seed)
    return x


def save_comps_x(path: str | Path, x: np.ndarray, comps: np.ndarray) -> Path:
    """最简导出：保存两列 `comp,x` 到文本文件。"""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        f.write("comp,x\n")
        for c, s in zip(comps.tolist(), x.tolist()):
            f.write(f"{int(c)},{s}\n")
    return out


if __name__ == "__main__":
    # 最小示例（仅演示函数调用）
    x, comps = sample_dx(n_bits=12, num_samples=1000, seed=7)
    out = save_comps_x("code/Train/data/dx_comps_x_demo1.txt", x=x, comps=comps)
    print("[demo] x, comps = sample_dx(12, 1000, seed=7)")
    print("[demo] x.shape =", x.shape)
    print("[demo] comps.shape =", comps.shape)
    print("[demo] output file =", str(out))
    print("[demo] first 5 samples =", x[:5].tolist())
