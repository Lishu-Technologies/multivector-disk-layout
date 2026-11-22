import argparse
import json
from pathlib import Path

import numpy as np

from .layout import fit_layout


def main():
    parser = argparse.ArgumentParser(description="Fit a static multivector SSD layout")
    parser.add_argument("tokens", type=Path, help="Concatenated FP16 token rows (.npy)")
    parser.add_argument("lengths", type=Path, help="Token count of each logical document (.npy)")
    parser.add_argument("shortlists", type=Path, help="Calibration candidate lists (.npy matrix or JSON array of arrays)")
    parser.add_argument("output", type=Path, help="New layout map (.npz)")
    args = parser.parse_args()
    if args.shortlists.suffix.lower() == ".json":
        history = json.loads(args.shortlists.read_text(encoding="utf-8"))
    else:
        history = np.load(args.shortlists, allow_pickle=False)
    layout = fit_layout(np.load(args.tokens, mmap_mode="r", allow_pickle=False),
                        np.load(args.lengths, allow_pickle=False), history)
    layout.save(args.output)
    print(f"Saved the ID map for {len(layout)} documents to {args.output}")


if __name__ == "__main__":
    main()
