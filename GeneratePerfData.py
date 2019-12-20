#!/usr/bin/env python3
"Generate test data for ONNX Runtime onnxruntime_perf_test tool."

import os
import argparse

import onnx
import onnx.numpy_helper
import onnxruntime
import numpy as np


def _main():

    parser = argparse.ArgumentParser("Generate random test input for ONNX model")
    parser.add_argument("--model", required=True, help="ONNX model file")
    parser.add_argument("--output", required=True, help="Output data directory")
    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.mkdir(args.output)

    sess = onnxruntime.InferenceSession(args.model)

    for inp in sess.get_inputs():
            # FIXME: allow user to specify omitted dimensions instead of always using 1
            shape = [s if isinstance(s, int) and s > 0 else 1 for s in inp.shape]
            # FIXME: use correct type based on inp.type instead of np.float32
            data = np.ones(shape, dtype=np.float32)
            tensor = onnx.numpy_helper.from_array(data, inp.name)
            path = os.path.join(args.output, inp.name + ".pb")
            print("%s: %s/%s %s" % (path, inp.type, data.dtype, data.shape))
            with open(path, 'wb') as outfile:
                outfile.write(tensor.SerializeToString())


if __name__ == "__main__":
    _main()
