from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
	# 2D data
	x2 = [0.03, 0.08, 0.2, 0.6, 1.8, 4.5, 12, 37, 100]
	y2a = [0.2, 0.18, 0.15, 0.13, 0.11, 0.09, 0.075, 0.06, 0.05]
	y2b = [0.39, 0.24, 0.15, 0.08, 0.05, 0.025, 0.018, 0.012, 0.007]
	y2c = [0.28, 0.23, 0.17, 0.14, 0.1, 0.08, 0.04, 0.025, 0.018]

	# 3D data
	x3 = [0.03, 0.08, 0.2, 0.6, 1.8]
	y3a = [0.026, 0.013, 0.006, 0.0031, 0.0022]
	y3b = [0.28, 0.08, 0.03, 0.009, 0.0025]
	y3c = [0.028, 0.013, 0.0045, 0.0025, 0.0012]

	output_dir = Path(__file__).resolve().parent / "assets"
	output_dir.mkdir(parents=True, exist_ok=True)
	output_file = output_dir / "1.png"

	plt.figure(figsize=(10, 6), dpi=160)

	# 2D curves
	plt.plot(x2, y2a, marker="o", linewidth=2, label="2D Y2a (three-body merger)")
	plt.plot(x2, y2b, marker="o", linewidth=2, label="2D Y2b (two-body merger)")
	plt.plot(x2, y2c, marker="o", linewidth=2, label="2D Y2c (e > 0.1)")

	# 3D curves
	plt.plot(x3, y3a, marker="s", linewidth=2, linestyle="--", label="3D Y3a (three-body merger)")
	plt.plot(x3, y3b, marker="s", linewidth=2, linestyle="--", label="3D Y3b (two-body merger)")
	plt.plot(x3, y3c, marker="s", linewidth=2, linestyle="--", label="3D Y3c (e > 0.1)")

	plt.xscale("log")
	plt.xlabel("Initial semi-major axis")
	plt.ylabel("Merger probability")
	plt.title("Merger Probability vs Initial Semi-major Axis (2D & 3D)")
	plt.grid(True, which="both", linestyle="--", alpha=0.45)
	plt.legend(fontsize=9, ncol=2)
	plt.tight_layout()
	plt.savefig(output_file, bbox_inches="tight")
	plt.close()

	print(f"Saved figure to: {output_file}")


if __name__ == "__main__":
	main()
