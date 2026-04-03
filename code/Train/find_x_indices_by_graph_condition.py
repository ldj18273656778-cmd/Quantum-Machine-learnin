from __future__ import annotations

"""Find sample indices satisfying a graph-local bit condition.

Condition for a target bit j:
1) x[j] == '0'
2) For all neighbors k connected to j in graph G, x[k] == '1'

Supports loading x-bitstrings from:
- .npy saved dict/object containing key 'x' (e.g., xy_dataset.npy)
- .txt/.csv containing x in the second column (e.g., comp,x or comp\tx\ty)
"""

from pathlib import Path
import sys
import csv
from typing import Iterable

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = ROOT / "code"

if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from sampling.ISQNN_generate_y import idqnn_connectivity


def load_x_bitstrings(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()

    if suffix == ".npy":
        obj = np.load(path, allow_pickle=True)
        data = obj.item() if isinstance(obj, np.ndarray) and obj.shape == () else obj#如果加载的对象是一个零维数组（标量），则使用 item() 方法获取其中的值，否则直接使用加载的对象。这种处理方式可以兼容两种情况：1) 直接保存了一个字典或其他对象；2) 保存了一个零维数组，其中包含了一个字典或其他对象。
        if isinstance(data, dict) and "x" in data:
            x = np.asarray(data["x"]).astype(str) #to array then to string array
            return x
        raise ValueError(f"{path} does not contain key 'x'.")

    if suffix in {".txt", ".csv"}:
        x_list: list[str] = []
        with path.open("r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="," if suffix == ".csv" else "\t")
            for row in reader:
                if not row:
                    continue
                joined = "".join(row).lower()
                if "n1=" in joined or "theta_path" in joined:
                    continue
                if row[0].lower() in {"comp", "n", "seed"}:
                    continue

                # Expected formats:
                # 1) comp,x
                # 2) comp\tx\ty
                if len(row) >= 2 and set(row[1].strip()).issubset({"0", "1"}):
                    x_list.append(row[1].strip())
                elif len(row) >= 1 and set(row[0].strip()).issubset({"0", "1"}):
                    x_list.append(row[0].strip())

        if not x_list:
            raise ValueError(f"No valid bitstrings found in {path}.")
        return np.asarray(x_list, dtype=str)

    raise ValueError(f"Unsupported file type: {path.suffix}")


def build_adjacency(n: int, edges: Iterable[tuple[int, int]]) -> list[set[int]]:#输入连通结构的边列表，输出邻接表，即每个节点的邻居集合。函数首先创建一个长度为 n 的列表，每个元素是一个空集合。然后遍历边列表，对于每条边 (a, b)，将 b 添加到 a 的邻居集合中，将 a 添加到 b 的邻居集合中。最后返回构建好的邻接表。
    adj: list[set[int]] = [set() for _ in range(n)]
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    return adj


def find_indices(
    x: np.ndarray,
    target_bit: int,
    adjacency: list[set[int]],
    show_progress: bool = True,
) -> np.ndarray:
    if x.ndim != 1 or len(x) == 0:
        raise ValueError("x must be a non-empty 1D array of bitstrings.")

    n = len(x[0])
    if not (0 <= target_bit < n):
        raise ValueError(f"target_bit out of range: {target_bit}, n={n}")

    neighbors = sorted(adjacency[target_bit])

    matched: list[int] = []
    iterator = enumerate(x.tolist())
    if show_progress and tqdm is not None:
        iterator = tqdm(iterator, total=len(x), desc="Finding matches", unit="sample")

    for idx, s in iterator:#idx是x中bitstring的索引，一共有N_samlpes个
        if len(s) != n:
            raise ValueError(f"Inconsistent bitstring length at index {idx}.")
        if s[target_bit] != "0":#如果目标位不是0，直接跳过这个样本
            continue
        if all(s[k] == "1" for k in neighbors):
            matched.append(idx)#把符合要求的索引添加到matched列表中

    return np.asarray(matched, dtype=int)


if __name__ == "__main__":
    # ===== 手动参数区 =====
    input_path = ROOT / "code" / "Train" / "data" / "xy_dataset.npy"
    output_path = ROOT / "code" / "Train" / "data" / "matched_indices.txt"

    n1 = 3
    m = 4

    # 目标位（人类编号，从1开始）
    target_bit_1based = 5
    # ====================

    x = load_x_bitstrings(Path(input_path))
    n = len(x[0])

    G = idqnn_connectivity(n1, m)
    if G["n"] != n:
        raise ValueError(f"Graph size mismatch: G.n={G['n']}, bitstring length={n}")

    target_bit = target_bit_1based - 1
    adjacency = build_adjacency(n=n, edges=G["all_edges"])#生成每个比特的连通邻居列表
    neighbors = sorted(adjacency[target_bit])

    indices = find_indices(x=x, target_bit=target_bit, adjacency=adjacency)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write(f"input={input_path}\n")
        f.write(f"n1={n1}, m={m}, n={n}\n")
        f.write(f"target_bit_1based={target_bit_1based}\n")
        f.write(f"target_bit_0based={target_bit}\n")
        f.write(f"neighbors_0based={neighbors}\n")
        f.write(f"count={len(indices)}\n")
        f.write("indices_0based:\n")
        for i in indices.tolist():
            f.write(f"{i}\n")

    print("=== Done ===")
    print(f"Input: {input_path}")
    print(f"Neighbors (0-based): {neighbors}")
    print(f"Matched count: {len(indices)}")
    print(f"Output: {output_path}")
    print(indices)
    print(adjacency[target_bit])
