#!/usr/bin/env python3
r"""
Generate test data for ONNX Runtime onnxruntime_perf_test tool.

Run:
    python.exe GeneratePerfData.py --model .\model\model.onnx --output .\model\data\

This call will create a directory ./model_data/ and produce protobuf files
containing data for ONNX model inputs, one file for each input.
The files are named `.\model\data\[model_input_name].pb`

After that, run the ONNX Runtime benchmark tool, e.g.

    onnxruntime_perf_test.exe -m times -r 1000 model\model.onnx model\data\

NOTE: ORT `onnxruntime_perf_test.exe` is very finicky about the model and data paths!
Both model .onnx file and data directory MUST reside in the same directory!
Also, on Windows it does not like the forward slash `/` as path separator.

For more details on the ONNX perf test tool, see
https://github.com/microsoft/onnxruntime/tree/master/onnxruntime/test/perftest

FIXME: The script currently breaks if some inputs have special characters like `/` or `:`
in their names (this is the case for models converted as is from TensorFlow).
TODO: generate random data instead of all 1s; allow user to specify the distribution.
"""

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
            # TODO: allow user to specify omitted dimensions instead of always using 1
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
