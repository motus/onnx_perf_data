#!/usr/bin/env python3

import argparse

import numpy as np
import onnxruntime as ort


def compare_models(models, tolerance=1.0e-4):

    sessions = [ort.InferenceSession(m) for m in models]

    inputs = {
        # FIXME: use the type from model instead of float32
        inp.name: np.float32(np.random.randn(
            *[s if isinstance(s, int) and s > 0 else 1 for s in inp.shape]))
        for inp in sessions[0].get_inputs()}

    outputs = [s.run(None, inputs) for s in sessions]

    match = True
    for i in range(len(outputs) - 1):
        for j in range(i + 1, len(outputs)):
            for (o1, o2) in zip(outputs[i], outputs[j]):
                if (np.abs(o1 - o2) > tolerance).any():
                    print("models %s and %s don't match" % (models[i], models[j]))
                    match = False

    if match:
        print("All models produce identical results!")


def _main():
    parser = argparse.ArgumentParser("Check if several ONNX models produce identical results")
    parser.add_argument("models", nargs="+", help="ONNX model files")
    parser.add_argument("--tolerance", type=float, default=1.0e-4,
                        help="Tolerance when comparing the models' outputs")
    args = parser.parse_args()
    compare_models(args.models, args.tolerance)


if __name__ == "__main__":
    _main()
