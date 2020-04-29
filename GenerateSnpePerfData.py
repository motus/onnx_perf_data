#!/usr/bin/env python3
r"""
Generate test data for Qualcomm SNPE benchmarking and quantization.

To run (or quantize) DLC model using the data, do e.g.

  snpe-net-run \
    --container model.dlc \
    --input_list input-list.txt

  snpe-dlc-quantize \
    --input_dlc model.dlc \
    --input_list input-list.txt

Where input-list.txt is from the --txt option of this script.
"""

import os
import argparse

import numpy as np
try:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
except:
    import tensorflow as tf


def _main():

    parser = argparse.ArgumentParser(
        description="Generate random test input for Qualcomm SNPE model")

    parser.add_argument("--model", required=True, help="TFLite model file")
    parser.add_argument("--samples", type=int, default=1, help="Number of samples to generate")
    parser.add_argument("--output", required=True, help="Output data directory")
    parser.add_argument("--txt", required=True,
                        help="File to save the list of input and output tensors to")

    args = parser.parse_args()

    interpreter = tf.lite.Interpreter(args.model)
    interpreter.allocate_tensors()

    if not os.path.exists(args.output):
        os.mkdir(args.output)

    with open(args.txt, "w", newline="\n") as txt_file:

        # txt_file.write("#%s\n" % " ".join(n["name"] for n in interpreter.get_output_details()))

        for n_sample in range(args.samples):

            out_dir = "%s/%03d" % (args.output, n_sample)
            if not os.path.exists(out_dir):
                os.mkdir(out_dir)

            snpe_inputs = []
            for (i, inp) in enumerate(interpreter.get_input_details()):
                data = inp["dtype"](np.random.randn(*inp["shape"]))
                path = "%s/input_%03d.raw" % (out_dir, i)
                print("%s: %s %s %s" % (path, inp["name"], data.dtype, data.shape))
                snpe_inputs.append("%s:=%s" % (inp["name"], path))
                with open(path, 'wb') as binary_file:
                    binary_file.write(data.tobytes())

            txt_file.write(" ".join(snpe_inputs) + "\n")


if __name__ == "__main__":
    _main()
