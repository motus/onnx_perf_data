#!/usr/bin/env python3

import argparse

import numpy as np
import coremltools


def compare_models(model_files, tolerance=1.0e-6):

    models = [coremltools.models.MLModel(fname) for fname in model_files]

    inputs = {
        # FIXME: use the type from model instead of float32
        inp.name: np.float32(np.random.randn(*inp.type.multiArrayType.shape))
        for inp in models[0].get_spec().description.input}

    outputs = [m.predict(inputs) for m in models]

    match = True
    for i in range(len(outputs) - 1):
        keys = outputs[i].keys()
        for j in range(i + 1, len(outputs)):
            for k in keys:
                (o1, o2) = (outputs[i][k], outputs[j][k])
                if (np.abs(o1 - o2) > tolerance).any():
                    print("models %s and %s don't match" % (model_files[i], model_files[j]))
                    match = False

    if match:
        print("All models produce identical results!")

    return match


def _main():
    parser = argparse.ArgumentParser("Check if several CoreML models produce identical results")
    parser.add_argument("models", nargs="+", help="CoreML model files")
    parser.add_argument("--tolerance", type=float, default=1.0e-6,
                        help="Tolerance when comparing the models' outputs")
    args = parser.parse_args()
    compare_models(args.models, args.tolerance)


if __name__ == "__main__":
    _main()
