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

import argparse
import numpy
try:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
except:
    import tensorflow as tf


def _numpy_from_file(fname, dtype, shape):
    return numpy.frombuffer(open(fname, "rb").read(), dtype=dtype).reshape(*shape)


def _main():

    parser = argparse.ArgumentParser(
        description="Compare the TFLite model output to the results"
                    " produced by the equlvalent Qualcomm SNPE model")
    parser.add_argument("--model", required=True, help="TFLite model file")
    parser.add_argument("--output_tensors", default=None,
                        help="Comma-separated list of output tensors")
    parser.add_argument("--output_parents", default=None,
                        help="Comma-separated list of output tensors' parents")
    parser.add_argument("--output_dir", required=True,
                        help="Path to raw SNPE output data")
    parser.add_argument("--input_list", required=True,
                        help="SNPE map of input tensors and data."
                             " (See `snpe-net-run --input_list`)")
    parser.add_argument("--stats", action="store_true",
                        help="Disable printing stats on values that don't match")
    parser.add_argument("--diff", type=int, default=0,
                        help="Print diff on first and last N values that don't match")
    parser.add_argument("--tolerance", type=float, default=1.0e-6,
                        help="Data matching tolerance")

    args = parser.parse_args()

    interpreter = tf.lite.Interpreter(args.model)
    interpreter.allocate_tensors()

    output_names = {}
    if args.output_tensors is not None:
        parents = args.output_parents or args.output_tensors
        output_names = {
            name.strip(): parent.strip()
            for (name, parent) in zip(args.output_tensors.split(","), parents.split(","))
        }

    i = 0
    for line in open(args.input_list):

        if len(line) == 0 or line[:1] == "#":
            continue

        inputs = dict(pair.strip().split(":=") for pair in line.split(" "))

        for inp in interpreter.get_input_details():
            fname = inputs[inp["name"] + ":0"]
            input_data = _numpy_from_file(fname, numpy.float32, inp['shape'])
            interpreter.set_tensor(inp['index'], input_data)

        interpreter.invoke()

        total_diff = 0
        total_size = 0
        for out in interpreter.get_output_details():
            name = out['name']
            fname = "%s/Result_%d/%s.raw" % (
                args.output_dir, i, output_names.get(name, name + ":0"))
            data = interpreter.get_tensor(out['index'])
            test_data = _numpy_from_file(fname, numpy.float32, out['shape'])

            diff = (numpy.abs(data - test_data) > args.tolerance).flatten().sum()
            total_diff += diff
            total_size += data.size

            if args.diff > 0:
                n = args.diff
                print("\n%smatch: %s %s: %s (%6.2f%% match)" % (
                    "NO " if diff > 0 else "", name, data.shape, fname,
                    100 - diff * 100.0 / data.size))
                print("  SNPE[:%d] : %s..." % (n, data.flatten()[:n]))
                print("TFLite[:%d] : %s..." % (n, test_data.flatten()[:n]))
                print("  SNPE[-%d:]: ...%s" % (n, data.flatten()[-n:]))
                print("TFLite[-%d:]: ...%s" % (n, test_data.flatten()[-n:]))

        if total_diff == 0:
            print("\n%3d: TFLite and SNPE outputs MATCH!" % i)
        else:
            print("\n%3d: TFLite and SNPE outputs DO NOT match!" % i)
            if args.stats:
                print("%d out of %d values (%.2f%%) DO NOT match!" % (
                    total_diff, total_size, total_diff * 100.0 / total_size))

        i += 1


if __name__ == "__main__":
    _main()
