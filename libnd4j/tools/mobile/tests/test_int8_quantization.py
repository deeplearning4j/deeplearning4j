# SPDX-License-Identifier: Apache-2.0

import copy
import importlib.util
import pathlib
import unittest

MODULE_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "validate-int8-quantization.py"
)
SPEC = importlib.util.spec_from_file_location("sdx_int8_validator", MODULE_PATH)
VALIDATOR = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(VALIDATOR)


def valid_contract():
    return {
        "formatVersion": 1,
        "scheme": "int8-per-channel",
        "provider": "sdx-graph",
        "targetSocs": ["Tensor_G5", "SM8750"],
        "deviceOnly": True,
        "allowFloatFallback": False,
        "requireVendorAot": True,
        "weights": {
            "dtype": "INT8",
            "scaleDtype": "FLOAT32",
            "granularity": "per-channel",
            "channelAxis": 0,
            "symmetric": True,
            "zeroPoint": 0,
        },
        "activations": {"dtype": "FLOAT16"},
        "excludedOps": [],
    }


class Int8QuantizationContractTest(unittest.TestCase):

    def test_accepts_fail_closed_weight_only_contract(self):
        summary = VALIDATOR.validate_contract(valid_contract())
        self.assertEqual("INT8", summary["weightDtype"])
        self.assertEqual("FLOAT16", summary["activationDtype"])
        self.assertFalse(summary["allowFloatFallback"])

    def test_rejects_float_fallback(self):
        contract = valid_contract()
        contract["allowFloatFallback"] = True
        with self.assertRaises(VALIDATOR.QuantizationContractError):
            VALIDATOR.validate_contract(contract)

    def test_rejects_asymmetric_or_per_tensor_weights(self):
        asymmetric = valid_contract()
        asymmetric["weights"]["zeroPoint"] = 4
        with self.assertRaises(VALIDATOR.QuantizationContractError):
            VALIDATOR.validate_contract(asymmetric)

        per_tensor = valid_contract()
        per_tensor["weights"]["granularity"] = "per-tensor"
        with self.assertRaises(VALIDATOR.QuantizationContractError):
            VALIDATOR.validate_contract(per_tensor)

    def test_requires_calibration_for_int8_activations(self):
        contract = valid_contract()
        contract["activations"] = {"dtype": "INT8"}
        with self.assertRaises(VALIDATOR.QuantizationContractError):
            VALIDATOR.validate_contract(contract)

        calibrated = copy.deepcopy(contract)
        calibrated["activations"]["calibration"] = {
            "method": "percentile",
            "percentile": 99.9,
            "sampleCount": 128,
            "datasetSha256": "a" * 64,
        }
        summary = VALIDATOR.validate_contract(calibrated)
        self.assertEqual("INT8", summary["activationDtype"])

    def test_rejects_unbound_calibration_dataset(self):
        contract = valid_contract()
        contract["activations"] = {
            "dtype": "INT8",
            "calibration": {
                "method": "minmax",
                "sampleCount": 128,
            },
        }
        with self.assertRaises(VALIDATOR.QuantizationContractError):
            VALIDATOR.validate_contract(contract)


if __name__ == "__main__":
    unittest.main()
