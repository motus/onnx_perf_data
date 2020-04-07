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


def _main(tolerance=1.0e-6):

    parser = argparse.ArgumentParser(
        description="Compare the ONNX model output to the results"
                    " produced by the equlvalent Qualcomm SNPE model")
    parser.add_argument("--model", required=True, help="ONNX model file")
    parser.add_argument("--output", required=True, help="Path to raw SNPE output data")
    parser.add_argument("--input_list", required=True,
                        help="SNPE map of input tensors and data."
                             " (See `snpe-net-run --input_list`)")

    args = parser.parse_args()

    sess = onnxruntime.InferenceSession(args.model)
    input_shapes = {n.name: n.shape for n in sess.get_inputs()}
    output_shapes = {n.name: n.shape for n in sess.get_outputs()}

    sample_num = 0
    output_names = None
    for line in open(args.input_list):

        if line[:1] == "#":
            output_names = line[1:].split(" ")
            continue

        inputs = {}
        for pair in line.split(" "):
            (name, fname) = pair.split(":=")
            inputs[name] = np.frombuffer(
                open(fname, "rb").read(),
                dtype=np.float32).reshape(input_shapes[name])

        outputs = sess.run(output_names or output_shapes.keys(), inputs)

        for (out_onnx, (name, shape)) in zip(outputs, output_shapes.items()):
            fname = "%s/Result_%d/%s.raw" % (args.output, sample_num, name)
            if os.path.exists(fname):
                out_snpe = np.frombuffer(
                    open(fname, "rb").read(), dtype=np.float32).reshape(shape)
                diff = (np.abs(out_onnx - out_snpe) > tolerance).flatten().sum()
                print("COMPARE: ", fname)
                print(diff)
            else:
                print("   SKIP: ", fname)

        sample_num += 1


if __name__ == "__main__":
    _main()
