#!/usr/bin/env python3

import argparse

import numpy as np
import onnxruntime
import coremltools


def compare_models(model_file_onnx, model_file_coreml,
                   tolerance=1.0e-6, print_stats=True, n_diff=5):
    "Check if ONNX and CoreML models produce identical results."

    model_onnx = onnxruntime.InferenceSession(model_file_onnx)
    model_coreml = coremltools.models.MLModel(model_file_coreml)

    inputs = {
        # FIXME: use the type from model instead of float32
        inp.name: np.float32(np.random.randn(
            *[s if isinstance(s, int) and s > 0 else 1 for s in inp.shape]))
        for inp in model_onnx.get_inputs()}

    outputs_onnx = model_onnx.run(None, inputs)  # list of outputs without names
    outputs_coreml = model_coreml.predict(inputs, useCPUOnly=True)  # a dictionary with names

    match = True
    for (out_onnx, (name, out_coreml)) in zip(outputs_onnx, outputs_coreml.items()):
        print("%s: ONNX %s:%s, CoreML %s:%s :: " % (
            name, out_onnx.dtype, out_onnx.shape, out_coreml.dtype, out_coreml.shape), end="")
        diff = (np.abs(out_onnx - out_coreml) > tolerance).flatten().sum()
        if diff:
            match = False
            if print_stats:
                print("%d out of %d values (%.2f%%) DO NOT match!" % (
                    diff, out_onnx.size, diff * 100.0 / out_onnx.size))
            if n_diff > 0:
                print("%s[:%d] DIFF ::\n  ONNX = %s\nCoreML = %s" % (
                    name, n_diff, out_onnx.flatten()[:n_diff], out_coreml.flatten()[:n_diff]))
                print("%s[-%d:] DIFF ::\n  ONNX = %s\nCoreML = %s" % (
                    name, n_diff, out_onnx.flatten()[-n_diff:], out_coreml.flatten()[-n_diff:]))
        else:
            print("Match!")

    return match


def _main():
    parser = argparse.ArgumentParser("Check if ONNX and CoreML models produce identical results")
    parser.add_argument("--onnx", required=True, help="ONNX model file")
    parser.add_argument("--coreml", required=True, help="CoreML model file")
    parser.add_argument("--tolerance", type=float, default=1.0e-6,
                        help="Tolerance when comparing the models' outputs")
    parser.add_argument("--no-stats", action="store_false",
                        help="Disable printing stats on values that don't match")
    parser.add_argument("--diff", type=int, default=0,
                        help="Print diff on first and last N values that don't match")
    args = parser.parse_args()
    match = compare_models(args.onnx, args.coreml, args.tolerance, args.no_stats, args.diff)
    print("Models %s match!" % {False: "DO NOT", True: "DO"}[match])


if __name__ == "__main__":
    _main()
