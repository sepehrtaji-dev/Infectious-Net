"""Verify there are no near-duplicates between train/val and test.

    python check_leakage.py
    python check_leakage.py --holdout Dataset_2
"""
import argparse

from Dataset import DUP_THRESHOLD, hash_all, make_splits


def source_of(path):
    return next((p for p in path.parts if p.lower().startswith("dataset")), "?")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="Data")
    ap.add_argument("--holdout", default=None)
    ap.add_argument("--threshold", type=int, default=DUP_THRESHOLD)
    args = ap.parse_args()

    train_s, val_s, test_s = make_splits(args.data, args.holdout)
    print(f"train={len(train_s)} val={len(val_s)} test={len(test_s)}")

    hashes = hash_all(train_s + val_s + test_s)
    seen = [(hashes[p], p) for p, _ in train_s + val_s]
    near = exact = 0
    examples = []
    for p, _ in test_s:
        h = hashes[p]
        best_d, best_p = 99, None
        for h2, p2 in seen:
            d = (h ^ h2).bit_count()
            if d < best_d:
                best_d, best_p = d, p2
        exact += best_d == 0
        if best_d <= args.threshold:
            near += 1
            if len(examples) < 5:
                examples.append((p, best_p, best_d))

    n = max(len(test_s), 1)
    print(f"\nExact matches in train/val : {exact}/{n} ({exact / n:.1%})")
    print(f"Near-duplicates (d<={args.threshold})   : {near}/{n} ({near / n:.1%})")
    for p, q, d in examples:
        print(f"  d={d}: {p}\n        ~ {q}")
    print("\nOK: no leakage detected." if near == 0 else "\nStill leaking - lower DUP_THRESHOLD issues? inspect examples.")


if __name__ == "__main__":
    main()