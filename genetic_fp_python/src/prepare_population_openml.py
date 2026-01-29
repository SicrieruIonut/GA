import argparse
from pathlib import Path
import numpy as np
import openml


def minmax_to_range(x: np.ndarray, out_min: float, out_max: float) -> np.ndarray:
    x_min = np.min(x, axis=0)
    x_max = np.max(x, axis=0)
    denom = np.where((x_max - x_min) == 0, 1.0, (x_max - x_min))
    scaled01 = (x - x_min) / denom
    return scaled01 * (out_max - out_min) + out_min


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--openml_id", type=int, default=42769)
    p.add_argument("--dim", type=int, default=10)
    p.add_argument("--out", type=str, default="data/population.npy")
    args = p.parse_args()

    print(f"Downloading OpenML dataset id={args.openml_id} ...")
    ds = openml.datasets.get_dataset(args.openml_id)

    X, y, categorical_indicator, attribute_names = ds.get_data(
        dataset_format="array", target=ds.default_target_attribute
    )

    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("Expected 2D feature matrix")

    if X.shape[1] < args.dim:
        raise ValueError(f"Dataset has {X.shape[1]} features but dim={args.dim}")

    X = X[:, : args.dim]
    X_scaled = minmax_to_range(X, -5.12, 5.12).astype(np.float32, copy=False)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, X_scaled)

    print("Saved:", out_path)
    print("Shape:", X_scaled.shape, "Dtype:", X_scaled.dtype)
    print("Done.")


if __name__ == "__main__":
    main()
