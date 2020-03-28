#!/usr/bin/env python3
r"""
Generate test data for Qualcomm SNPE benchmarking and quantization.

To run DLC model with the data, do e.g.

  snpe-net-run \
      --container model.dlc \
      --input_list input-list.txt

Where input-list.txt is from the --txt option of this script.
"""

import os
import argparse

import onnxruntime
import numpy as np


def _main():

    parser = argparse.ArgumentParser(description="Generate random test input for ONNX model")
    parser.add_argument("--model", required=True, help="ONNX model file")
    parser.add_argument("--output", required=True, help="Output data directory")
    parser.add_argument("--txt", required=True,
                        help="File to save the list of input and output tensors to")

    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.mkdir(args.output)

    sess = onnxruntime.InferenceSession(args.model)

    snpe_inputs = []
    for (i, inp) in enumerate(sess.get_inputs()):
        # FIXME: use correct type based on inp.type instead of np.float32
        data = np.float32(np.random.randn(*inp.shape))
        path = "%s/input_%03d.raw" % (args.output, i)
        print("%s: %s %s/%s %s" % (path, inp.name, inp.type, data.dtype, data.shape))
        snpe_inputs.append("%s:=%s" % (inp.name, path))
        with open(path, 'wb') as outfile:
            outfile.write(data.tobytes())

    with open(args.txt, "w", newline="\n") as outfile:
        # outfile.write("#%s\n" % " ".join(n.name for n in sess.get_outputs()))
        outfile.write(" ".join(snpe_inputs) + "\n")


if __name__ == "__main__":
    _main()
