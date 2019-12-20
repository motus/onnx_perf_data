#!/usr/bin/env python3
"Generate test data for ONNX Runtime onnxruntime_perf_test tool."

import argparse

import onnx
import onnx.numpy_helper
import onnxruntime
import numpy as np


def _main():

    parser = argparse.ArgumentParser("Generate random test input for ONNX model")
    parser.add_argument("--model", required=True, help="ONNX model file")
    parser.add_argument("--output", default=None,
                        help="Output protobuf file. Default is [model].pb")
    args = parser.parse_args()

    output_path = args.output
    if output_path is None:
        output_path = args.model[:-4] + "pb"  # replace .onnx with .pb

    print("Output:", output_path)
    with open(output_path, 'wb') as outfile:
        sess = onnxruntime.InferenceSession(args.model)
        for inp in sess.get_inputs():
            shape = [s if isinstance(s, int) and s > 0 else 1 for s in inp.shape]
            data = np.ones(shape, dtype=np.float32)
            print("    %s %s %s" % (inp.name, inp.type, shape))
            tensor = onnx.numpy_helper.from_array(data, inp.name)
            outfile.write(tensor.SerializeToString())


if __name__ == "__main__":
    _main()
