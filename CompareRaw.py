#!/usr/bin/env python3

import argparse
import numpy as np


def compare_data(fname1, fname2,
                 tolerance=1.0e-6, print_stats=True, n_diff=5):
    "Compare two raw numpy files and print the diff."
    print("file1 = %s\nfile2 = %s" % (fname1, fname2))
    out1 = np.frombuffer(open(fname1, "rb").read(), dtype=np.float32)
    out2 = np.frombuffer(open(fname2, "rb").read(), dtype=np.float32)
    diff = (np.abs(out1 - out2) > tolerance).sum()
    if diff:
        if print_stats:
            print("%d out of %d values (%.2f%%) DO NOT match!" % (
                diff, out1.size, diff * 100.0 / out1.size))
        if n_diff > 0:
            print(" [:%d] DIFF ::\nfile1 = %s\nfile2 = %s" % (
                n_diff, out1[:n_diff], out2[:n_diff]))
            print("[-%d:] DIFF ::\nfile1 = %s\nfile2 = %s" % (
                n_diff, out1[-n_diff:], out2[-n_diff:]))
    return not diff


def _main():
    parser = argparse.ArgumentParser(
        description="Check if ONNX and CoreML models produce identical results")
    parser.add_argument("file1", help="Input file as raw numpy bytes (assume float32)")
    parser.add_argument("file2", help="Input file as raw numpy bytes (assume float32)")
    parser.add_argument("--tolerance", type=float, default=1.0e-6,
                        help="Tolerance when comparing the models' outputs")
    parser.add_argument("--no-stats", action="store_false",
                        help="Disable printing stats on values that don't match")
    parser.add_argument("--diff", type=int, default=0,
                        help="Print diff on first and last N values that don't match")
    args = parser.parse_args()
    match = compare_data(args.file1, args.file2, args.tolerance, args.no_stats, args.diff)
    print("Files %s match!" % {False: "DO NOT", True: "DO"}[match])


if __name__ == "__main__":
    _main()
