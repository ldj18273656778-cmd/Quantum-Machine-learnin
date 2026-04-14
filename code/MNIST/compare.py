from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
MNIST_DIR = Path(__file__).resolve().parent


def main() -> None:
	input_path = MNIST_DIR / "data" / "y_inferred.npz"
	if not input_path.exists():
		raise FileNotFoundError(f"未找到文件: {input_path}")

	with np.load(input_path) as data:
		if "y_inferred" not in data:
			raise KeyError(f"{input_path} 中不存在键 y_inferred")
		if "y_test_encoded" not in data:
			raise KeyError(f"{input_path} 中不存在键 y_test_encoded")
		y_inferred = data["y_inferred"]
		y_test_encoded = data["y_test_encoded"]

	if y_inferred.ndim == 2 and y_inferred.shape[1] == 100:
		images_inferred = y_inferred.reshape(-1, 10, 10)
	elif y_inferred.ndim == 3 and y_inferred.shape[1:] == (10, 10):
		images_inferred = y_inferred
	else:
		raise ValueError(f"y_inferred 形状不支持: {y_inferred.shape}")

	if y_test_encoded.ndim == 2 and y_test_encoded.shape[1] == 100:
		images_encoded = y_test_encoded.reshape(-1, 10, 10)
	elif y_test_encoded.ndim == 3 and y_test_encoded.shape[1:] == (10, 10):
		images_encoded = y_test_encoded
	else:
		raise ValueError(f"y_test_encoded 形状不支持: {y_test_encoded.shape}")

	n_show = min(10, len(images_inferred), len(images_encoded))
	fig, axes = plt.subplots(n_show, 2, figsize=(6, 2 * n_show))

	if n_show == 1:
		axes = np.array([axes])

	for i in range(n_show):
		axes[i, 0].imshow(images_inferred[i], cmap="gray", vmin=0, vmax=1)
		axes[i, 0].set_title(f"inferred #{i}")
		axes[i, 0].axis("off")

		axes[i, 1].imshow(images_encoded[i], cmap="gray", vmin=0, vmax=1)
		axes[i, 1].set_title(f"encoded #{i}")
		axes[i, 1].axis("off")

	plt.tight_layout()
	plt.show()

	print(f"Displayed {n_show} pairs of inferred vs encoded images")


if __name__ == "__main__":
	main()
