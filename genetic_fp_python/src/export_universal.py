import argparse
import json
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npy", required=True)
    p.add_argument("--out_bin", required=True)
    args = p.parse_args()

    npy_path = Path(args.npy)
    out_bin = Path(args.out_bin)
    out_bin.parent.mkdir(parents=True, exist_ok=True)

    arr = np.load(npy_path, mmap_mode="r")
    if arr.ndim != 2:
        raise ValueError("Expected 2D array (N, D).")

    m = np.asarray(arr, dtype=np.float32, order="C")

    with open(out_bin, "wb") as f:
        for i in range(m.shape[0]):
            f.write(m[i].tobytes(order="C"))

    meta = {
        "file": out_bin.name,
        "shape": [int(m.shape[0]), int(m.shape[1])],
        "dtype": "float32",
        "order": "C_row_major",
        "endianness": "little",
    }
    out_json = out_bin.with_suffix(".json")
    out_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("Wrote:", out_bin)
    print("Meta :", out_json)


if __name__ == "__main__":
    main()
