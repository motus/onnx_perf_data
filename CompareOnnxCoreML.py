#!/usr/bin/env python3

import argparse

import numpy as np
import onnxruntime
import coremltools


def compare_models(model_file_onnx, model_file_coreml, tolerance=1.0e-4):
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

    for (out_onnx, out_coreml) in zip(outputs_onnx, outputs_coreml.values()):
        if (np.abs(out_onnx - out_coreml) > tolerance).any():
            return False

    return True


def _main():
    parser = argparse.ArgumentParser("Check if ONNX and CoreML models produce identical results")
    parser.add_argument("--onnx", required=True, help="ONNX model file")
    parser.add_argument("--coreml", required=True, help="CoreML model file")
    parser.add_argument("--tolerance", type=float, default=1.0e-4,
                        help="Tolerance when comparing the models' outputs")
    args = parser.parse_args()
    match = compare_models(args.onnx, args.coreml, args.tolerance)
    print("Models %s match!" % ["DO", "DO NOT"][int(match)])


if __name__ == "__main__":
    _main()
