import os
import argparse

# 修复部分 Windows/Conda 环境下 PyTorch 与 MKL 的 OpenMP 冲突
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
from torchvision import datasets, transforms

def binarize(img):
    return (img > 0.183).float()


def get_n_from_args_or_input(total: int) -> int:
	"""获取用户输入的第 n 张图（1-based），并返回对应 0-based 索引。"""
	parser = argparse.ArgumentParser(description="查看 MNIST 训练集的第 n 张图片")
	parser.add_argument("-n", "--num", type=int, help="要查看的图片序号（从 1 开始）")
	args = parser.parse_args()

	n = args.num
	if n is None:
		raw = input(f"请输入要查看的图片序号 n（1~{total}）: ").strip()
		if not raw.isdigit():
			raise ValueError("n 必须是正整数")
		n = int(raw)

	if n < 1 or n > total:
		raise ValueError(f"n 超出范围，应在 1~{total} 之间")

	return n - 1


def main() -> None:
	# transform = transforms.ToTensor()
	transform = transforms.Compose([
    transforms.Resize((10, 10)),  # 28x28 -> 10x10
    transforms.ToTensor(),       # [0,255] -> [0,1] 张量
	transforms.Lambda(binarize)
])

	train_dataset = datasets.MNIST(
		root="./data",
		train=True,
		download=True,
		transform=transform,
	)

	print("train size:", len(train_dataset))
	total = len(train_dataset)
	idx = get_n_from_args_or_input(total)
	img, label = train_dataset[idx]

	plt.figure(figsize=(3, 3))
	plt.imshow(img.squeeze(0), cmap="gray")
	plt.title(f"MNIST train 第 {idx + 1} 张, label={label}")
	plt.axis("off")
	plt.tight_layout()
	plt.show()


if __name__ == "__main__":
	 main()
# 	n=16
# get_n_from_args_or_input(n)