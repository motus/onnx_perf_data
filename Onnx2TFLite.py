#!/usr/bin/env python3
"Functions to convert ONNX mdoel file to TFLite and to test TFLite models on random data"

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


if __name__ == "__main__":
    fname_onnx = "devel/models/Phasen/version201/opset10/trained.onnx"
    fname_tf = "devel/models/Phasen/version201/opset10/trained_tf.pb"
    fname_tflite = "devel/models/Phasen/version201/opset10/trained_tf.tflite"
    convert(fname_onnx, fname_tf, fname_tflite)
    test(fname_tflite)
