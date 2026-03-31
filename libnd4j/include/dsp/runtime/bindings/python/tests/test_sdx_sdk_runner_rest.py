# ******************************************************************************
#
# This program and the accompanying materials are made available under the
# terms of the Apache License, Version 2.0 which is available at
# https://www.apache.org/licenses/LICENSE-2.0.
#
# SPDX-License-Identifier: Apache-2.0
# ******************************************************************************

import json
import pathlib
import sys
import unittest

import numpy as np

_THIS_DIR = pathlib.Path(__file__).resolve().parent
_PARENT = _THIS_DIR.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

from fastapi.testclient import TestClient

from sdx_sdk_runner import SdxModelRegistry, create_app
from sdx_tensor_transport import (
    dtype_code_to_numpy,
    decode_npz_inputs,
    encode_npz_tensors,
    tensors_from_json_payload,
    tensors_to_json_payload,
)


class _FakeRegistry:
    def __init__(self):
        self._models = {}
        self._counter = 0
        self.last_run_inputs = []

    def abi_version(self):
        return 1

    def load_model(self, model_path, model_options=None, requested_outputs=None):
        self._counter += 1
        model_id = f"m{self._counter}"
        self._models[model_id] = {
            "model_path": model_path,
            "model_options": model_options,
            "requested_outputs": list(requested_outputs or []),
        }
        return model_id

    def unload_model(self, model_id):
        return self._models.pop(model_id, None) is not None

    def run(self, model_id, inputs, output_specs, run_options=None):
        if model_id not in self._models:
            raise KeyError(model_id)

        self.last_run_inputs = [name for name, _ in inputs]

        outputs = []
        for spec in output_specs:
            arr = np.arange(
                np.prod(spec.shape, dtype=np.int64),
                dtype=dtype_code_to_numpy(spec.dtype),
            ).reshape(spec.shape)
            outputs.append((spec.name, arr))

        report = {
            "requested_backend": 0,
            "applied_backend": 0,
            "status_code": 0,
            "used_fallback": False,
            "execution_time_ns": 123,
            "requested_gpu_target": 0,
            "applied_gpu_target": 0,
        }
        return outputs, report


class SdxSdkRunnerRestTests(unittest.TestCase):
    def setUp(self):
        self.registry = _FakeRegistry()
        app = create_app(self.registry)
        self.client = TestClient(app)

    def _load_model(self):
        response = self.client.post(
            "/v1/models:load",
            json={
                "model_path": "/tmp/mock-model.sdz",
                "requested_outputs": ["output_0"],
            },
        )
        self.assertEqual(response.status_code, 200)
        return response.json()["model_id"]

    def test_health_endpoint(self):
        response = self.client.get("/healthz")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["abi_version"], 1)

    def test_load_and_unload_model(self):
        model_id = self._load_model()

        response = self.client.post(f"/v1/models/{model_id}:unload")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "unloaded")

    def test_run_json(self):
        model_id = self._load_model()

        input_arr = np.arange(4, dtype=np.float32).reshape(1, 4)
        request_payload = {
            "inputs": tensors_to_json_payload([("input_0", input_arr)]),
            "outputs": [{"name": "output_0", "dtype": 5, "shape": [1, 4]}],
        }

        response = self.client.post(f"/v1/models/{model_id}:run", json=request_payload)
        self.assertEqual(response.status_code, 200)

        body = response.json()
        outputs = tensors_from_json_payload(body["outputs"])
        self.assertEqual(outputs[0][0], "output_0")
        np.testing.assert_array_equal(outputs[0][1], np.arange(4, dtype=np.float32).reshape(1, 4))

    def test_run_npz_respects_input_order_header(self):
        model_id = self._load_model()

        npz_body = encode_npz_tensors(
            [
                ("mask", np.array([[1, 1]], dtype=np.int32)),
                ("token_ids", np.array([[7, 8]], dtype=np.int32)),
            ]
        )

        output_specs = [{"name": "output_0", "dtype": 9, "shape": [1, 2]}]
        input_order = ["token_ids", "mask"]

        response = self.client.post(
            f"/v1/models/{model_id}:run-npz",
            content=npz_body,
            headers={
                "Content-Type": "application/x-sdx-npz",
                "X-SDX-Output-Specs": json.dumps(output_specs),
                "X-SDX-Input-Order": json.dumps(input_order),
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.registry.last_run_inputs, input_order)

        decoded_outputs = decode_npz_inputs(response.content)
        self.assertEqual(decoded_outputs[0][0], "output_0")
        np.testing.assert_array_equal(decoded_outputs[0][1], np.array([[0, 1]], dtype=np.int32))

        report = json.loads(response.headers["X-SDX-Execution-Report"])
        self.assertEqual(report["status_code"], 0)

    def test_run_unknown_model_returns_not_found(self):
        response = self.client.post(
            "/v1/models/missing:run",
            json={
                "inputs": tensors_to_json_payload([("input_0", np.array([1], dtype=np.float32))]),
                "outputs": [{"name": "output_0", "dtype": 5, "shape": [1]}],
            },
        )
        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
