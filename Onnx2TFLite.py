#!/usr/bin/env python3
"Functions to convert ONNX mdoel file to TFLite and to test TFLite models on random data"

import os
import argparse

import numpy
import onnx
import onnx_tf
import tensorflow as tf


def convert(fname_onnx, fname_tf, fname_tflite):
    "Reead an ONNX model from a file and convert it to TensorFlow and TensorFlow Lite models"

    model_onnx = onnx.load(fname_onnx)

    model_prep_tf = onnx_tf.backend.prepare(model_onnx)
    model_prep_tf.export_graph(fname_tf)

    model_inputs = ["input"]
    model_outputs = ["output"]

    converter = tf.lite.TFLiteConverter.from_frozen_graph(
        fname_tf, model_inputs, model_outputs)
    tflite_model = converter.convert()

    with open(fname_tflite, "wb") as file_tflite:
        file_tflite.write(tflite_model)


def test(fname_tflite):
    "Read a TensorFlow lite model froam a file and evaluate it with some random input"

    interpreter = tf.lite.Interpreter(fname_tflite)
    interpreter.allocate_tensors()

    for inp in interpreter.get_input_details():
        input_data = numpy.float32(numpy.random.randn(*inp['shape']))
        interpreter.set_tensor(inp['index'], input_data)

    interpreter.invoke()

    return {out['name']: interpreter.get_tensor(out['index'])
            for out in interpreter.get_output_details()}


def _main():

    parser = argparse.ArgumentParser("Convert ONNX model to TF and TFLite")
    parser.add_argument("--onnx", required=True, help="Input ONNX model file")
    parser.add_argument("--tf", default=None, help="Output TensorFlow model file")
    parser.add_argument("--tflite", default=None, help="Output TensorFlow Lite model file")
    args = parser.parse_args()

    fname_tf = args.tf
    if not fname_tf:
        fname_tf = os.path.splitext(args.onnx)[0] + ".pb"

    fname_tflite = args.tflite
    if not fname_tflite:
        fname_tflite = os.path.splitext(fname_tf)[0] + ".tflite"

    convert(args.onnx, fname_tf, fname_tflite)
    print("\nSuccess! Converted:\n  ONNX: %s\n to TF: %s\nTFLite: %s\n"
          % (args.onnx, fname_tf, fname_tflite))

    print("Testing TFLite inference...")
    outputs = test(fname_tflite)
    print("\nTFLite inference tested! Output shapes:\n",
          {key: val.shape for (key, val) in outputs.items()})


if __name__ == "__main__":
    _main()
