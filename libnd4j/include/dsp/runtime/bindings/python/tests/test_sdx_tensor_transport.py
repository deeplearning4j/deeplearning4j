# ******************************************************************************
#
# This program and the accompanying materials are made available under the
# terms of the Apache License, Version 2.0 which is available at
# https://www.apache.org/licenses/LICENSE-2.0.
#
# SPDX-License-Identifier: Apache-2.0
# ******************************************************************************

import pathlib
import sys
import unittest

import numpy as np

_THIS_DIR = pathlib.Path(__file__).resolve().parent
_PARENT = _THIS_DIR.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

from sdx_tensor_transport import (
    decode_npz_inputs,
    encode_npz_tensors,
    order_named_tensors,
    parse_output_specs,
    tensor_from_bytes,
    tensor_to_bytes,
    tensors_from_json_payload,
    tensors_to_json_payload,
)


class SdxTensorTransportTests(unittest.TestCase):
    def test_json_round_trip(self):
        x = np.arange(6, dtype=np.float32).reshape(2, 3)
        y = np.arange(4, dtype=np.int32)

        payload = tensors_to_json_payload([("input_0", x), ("input_1", y)])
        decoded = tensors_from_json_payload(payload)

        self.assertEqual([name for name, _ in decoded], ["input_0", "input_1"])
        np.testing.assert_array_equal(decoded[0][1], x)
        np.testing.assert_array_equal(decoded[1][1], y)

    def test_tensor_from_bytes_validates_size(self):
        x = np.arange(4, dtype=np.float32)
        raw = tensor_to_bytes(x)

        with self.assertRaisesRegex(ValueError, "byte size mismatch"):
            tensor_from_bytes("bad", dtype=5, shape=[5], payload=raw)

    def test_parse_output_specs_rejects_bad_shape(self):
        with self.assertRaisesRegex(ValueError, "shape must be an array"):
            parse_output_specs([{"name": "out", "dtype": 5, "shape": "1,2"}])

    def test_decode_npz_orders_input_suffixes_numerically(self):
        blob = encode_npz_tensors(
            [
                ("input_10", np.array([10], dtype=np.int32)),
                ("input_2", np.array([2], dtype=np.int32)),
                ("input_1", np.array([1], dtype=np.int32)),
            ]
        )

        decoded = decode_npz_inputs(blob)
        self.assertEqual([name for name, _ in decoded], ["input_1", "input_2", "input_10"])

    def test_order_named_tensors_enforces_explicit_order(self):
        tensors = [
            ("token_ids", np.array([[1, 2]], dtype=np.int32)),
            ("mask", np.array([[1, 1]], dtype=np.int32)),
        ]

        ordered = order_named_tensors(tensors, ["mask", "token_ids"])
        self.assertEqual([name for name, _ in ordered], ["mask", "token_ids"])

        with self.assertRaisesRegex(ValueError, "not found"):
            order_named_tensors(tensors, ["missing", "token_ids"])

        with self.assertRaisesRegex(ValueError, "not listed"):
            order_named_tensors(tensors, ["token_ids"])


if __name__ == "__main__":
    unittest.main()
