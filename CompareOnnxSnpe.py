#!/usr/bin/env python3
r"""
Compare the data generated by for Qualcomm SNPE model
to the output of the equivalent ONNX model.

To produce the data with the DLC model, do e.g.

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

    parser = argparse.ArgumentParser(
        description="Compare the ONNX model output to the results"
                    " produced by the equlvalent Qualcomm SNPE model")
    parser.add_argument("--model", required=True, help="ONNX model file")
    parser.add_argument("--output_names", default=None,
                        help="Comma-separated list of output tensors")
    parser.add_argument("--output_data", required=True,
                        help="Path to raw SNPE output data")
    parser.add_argument("--input_list", required=True,
                        help="SNPE map of input tensors and data."
                             " (See `snpe-net-run --input_list`)")
    parser.add_argument("--tolerance", type=float, default=1.0e-6,
                        help="Data matching tolerance")

    args = parser.parse_args()

    sess = onnxruntime.InferenceSession(args.model)
    input_shapes = {n.name: n.shape for n in sess.get_inputs()}
    output_shapes = {n.name: n.shape for n in sess.get_outputs()}

    sample_num = 0
    output_names = args.output_names.split(",") if args.output_names else None
    for line in open(args.input_list):

        if line[:1] == "#" and output_names is None:
            output_names = line[1:].strip().split(" ")
            continue

        inputs = {}
        for pair in line.strip().split(" "):
            (name, fname) = pair.split(":=")
            inputs[name] = np.frombuffer(
                open(fname.strip(), "rb").read(),
                dtype=np.float32).reshape(input_shapes[name])

        outputs = sess.run(list(output_names or output_shapes), inputs)

        for (out_onnx, (name, shape)) in zip(outputs, output_shapes.items()):
            fname = "%s/Result_%d/%s.raw" % (args.output, sample_num, name)
            if os.path.exists(fname):
                out_snpe = np.frombuffer(
                    open(fname, "rb").read(), dtype=np.float32).reshape(shape)
                diff = (np.abs(out_onnx - out_snpe) > args.tolerance).flatten().sum()
                print("COMPARE: %f :: %6.2f%% of data match"
                      % (fname, diff * 100.0 / out_onnx.size))
            else:
                print("File not found: ", fname)

        sample_num += 1


if __name__ == "__main__":
    _main()