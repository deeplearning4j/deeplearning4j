#!/usr/bin/env python3
"""Validate the fail-closed INT8 contract used by SDX mobile AOT bundles."""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from typing import Any, Mapping, Sequence

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ALLOWED_ACTIVATION_DTYPES = {"FLOAT16", "FLOAT32", "INT8"}
_ALLOWED_CALIBRATION_METHODS = {"minmax", "percentile", "entropy"}
_ALLOWED_PROVIDERS = {"sdx-graph", "litert-lm"}


class QuantizationContractError(ValueError):
    """Raised when a quantization contract could enable an unsafe artifact."""


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise QuantizationContractError(f"{field} must be an object")
    return value


def _non_empty_strings(value: Any, field: str) -> Sequence[str]:
    if not isinstance(value, list) or not value:
        raise QuantizationContractError(
            f"{field} must be a non-empty string array")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise QuantizationContractError(
            f"{field} must contain only non-empty strings")
    return value


def _optional_sha256(value: Any, field: str) -> None:
    if value is None:
        return
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise QuantizationContractError(
            f"{field} must be a lowercase SHA-256 digest")


def validate_contract(data: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate and return a compact provider/AOT summary."""

    if data.get("formatVersion") != 1:
        raise QuantizationContractError("formatVersion must be 1")
    if data.get("scheme") != "int8-per-channel":
        raise QuantizationContractError(
            "scheme must be int8-per-channel")
    if data.get("provider") not in _ALLOWED_PROVIDERS:
        raise QuantizationContractError(
            "provider must be sdx-graph or litert-lm")
    if data.get("deviceOnly") is not True:
        raise QuantizationContractError("deviceOnly must be true")
    if data.get("allowFloatFallback") is not False:
        raise QuantizationContractError(
            "allowFloatFallback must be false")
    if data.get("requireVendorAot") is not True:
        raise QuantizationContractError(
            "requireVendorAot must be true")

    target_socs = _non_empty_strings(data.get("targetSocs"), "targetSocs")
    _optional_sha256(data.get("sourceModelSha256"), "sourceModelSha256")
    _optional_sha256(data.get("aotArtifactSha256"), "aotArtifactSha256")

    weights = _mapping(data.get("weights"), "weights")
    if weights.get("dtype") != "INT8":
        raise QuantizationContractError("weights.dtype must be INT8")
    if weights.get("scaleDtype") != "FLOAT32":
        raise QuantizationContractError(
            "weights.scaleDtype must be FLOAT32")
    if weights.get("granularity") != "per-channel":
        raise QuantizationContractError(
            "weights.granularity must be per-channel")
    if not isinstance(weights.get("channelAxis"), int):
        raise QuantizationContractError(
            "weights.channelAxis must be an integer")
    if weights.get("symmetric") is not True:
        raise QuantizationContractError(
            "weights.symmetric must be true")
    if weights.get("zeroPoint") != 0:
        raise QuantizationContractError(
            "symmetric INT8 weights require zeroPoint=0")

    activations = _mapping(data.get("activations"), "activations")
    activation_dtype = activations.get("dtype")
    if activation_dtype not in _ALLOWED_ACTIVATION_DTYPES:
        raise QuantizationContractError(
            "activations.dtype must be FLOAT16, FLOAT32, or INT8")

    calibration = activations.get("calibration")
    if activation_dtype == "INT8":
        calibration = _mapping(calibration, "activations.calibration")
        method = calibration.get("method")
        if method not in _ALLOWED_CALIBRATION_METHODS:
            raise QuantizationContractError(
                "calibration.method must be minmax, percentile, or entropy")
        sample_count = calibration.get("sampleCount")
        if not isinstance(sample_count, int) or sample_count < 32:
            raise QuantizationContractError(
                "calibration.sampleCount must be at least 32")
        _optional_sha256(
            calibration.get("datasetSha256"),
            "activations.calibration.datasetSha256")
        if calibration.get("datasetSha256") is None:
            raise QuantizationContractError(
                "INT8 activation calibration requires datasetSha256")
        if method == "percentile":
            percentile = calibration.get("percentile")
            if (not isinstance(percentile, (int, float))
                    or not 90.0 <= float(percentile) < 100.0):
                raise QuantizationContractError(
                    "percentile calibration requires percentile in [90, 100)")
    elif calibration is not None:
        raise QuantizationContractError(
            "calibration is only valid for INT8 activations")

    excluded_ops = data.get("excludedOps", [])
    if not isinstance(excluded_ops, list) or any(
            not isinstance(op, str) or not op.strip() for op in excluded_ops):
        raise QuantizationContractError(
            "excludedOps must be a string array")

    return {
        "formatVersion": 1,
        "scheme": "int8-per-channel",
        "provider": data["provider"],
        "targetSocs": list(target_socs),
        "weightDtype": "INT8",
        "activationDtype": activation_dtype,
        "deviceOnly": True,
        "allowFloatFallback": False,
        "requireVendorAot": True,
    }


def load_and_validate(path: pathlib.Path) -> Mapping[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise QuantizationContractError(
            f"cannot read quantization contract {path}: {error}") from error
    if not isinstance(data, dict):
        raise QuantizationContractError(
            "quantization contract root must be an object")
    return validate_contract(data)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate an SDX mobile INT8 per-channel contract")
    parser.add_argument("config", type=pathlib.Path)
    parser.add_argument(
        "--summary-json",
        action="store_true",
        help="print the normalized contract summary as JSON")
    args = parser.parse_args(argv)

    try:
        summary = load_and_validate(args.config)
    except QuantizationContractError as error:
        print(f"Invalid INT8 quantization contract: {error}", file=sys.stderr)
        return 1

    if args.summary_json:
        print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
    else:
        print(
            "Validated INT8 per-channel AOT contract: "
            f"provider={summary['provider']} "
            f"targets={','.join(summary['targetSocs'])} "
            f"activations={summary['activationDtype']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
