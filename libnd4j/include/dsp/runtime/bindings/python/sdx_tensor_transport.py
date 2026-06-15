"""Shared tensor transport utilities for SDX SDK serving.

This module centralizes ndarray encoding/decoding rules used by both REST and gRPC
runners so they stay behaviorally identical.
"""

from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple, Union

import numpy as np


# SDX runtime dtype numeric IDs match nd4j DataType::toInt() values.
_DTYPE_CODE_TO_NUMPY: Dict[int, np.dtype] = {
    1: np.dtype(np.bool_),
    3: np.dtype(np.float16),
    5: np.dtype(np.float32),
    6: np.dtype(np.float64),
    7: np.dtype(np.int8),
    8: np.dtype(np.int16),
    9: np.dtype(np.int32),
    10: np.dtype(np.int64),
    11: np.dtype(np.uint8),
    12: np.dtype(np.uint16),
    13: np.dtype(np.uint32),
    14: np.dtype(np.uint64),
}
if hasattr(np, "bfloat16"):
    _DTYPE_CODE_TO_NUMPY[17] = np.dtype(np.bfloat16)

_NUMPY_TO_DTYPE_CODE: Dict[np.dtype, int] = {v: k for k, v in _DTYPE_CODE_TO_NUMPY.items()}


@dataclass(frozen=True)
class TensorSpec:
    name: str
    dtype: int
    shape: Tuple[int, ...]


def dtype_code_to_numpy(dtype_code: int) -> np.dtype:
    dt = _DTYPE_CODE_TO_NUMPY.get(int(dtype_code))
    if dt is None:
        raise TypeError(f"Unsupported SDX dtype code: {dtype_code}")
    return dt


def numpy_dtype_to_code(dtype: np.dtype) -> int:
    code = _NUMPY_TO_DTYPE_CODE.get(np.dtype(dtype))
    if code is None:
        raise TypeError(f"Unsupported NumPy dtype for SDX: {dtype}")
    return code


def normalize_shape(shape: Sequence[int]) -> Tuple[int, ...]:
    out: List[int] = []
    for dim in shape:
        value = int(dim)
        if value < 0:
            raise ValueError(f"Shape dimensions must be >= 0, got {value}")
        out.append(value)
    return tuple(out)


def tensor_from_bytes(name: str, dtype: int, shape: Sequence[int], payload: bytes) -> np.ndarray:
    np_dtype = dtype_code_to_numpy(dtype)
    normalized_shape = normalize_shape(shape)
    expected_bytes = int(np.prod(normalized_shape, dtype=np.int64)) * np_dtype.itemsize
    if len(payload) != expected_bytes:
        raise ValueError(
            f"Tensor '{name}' byte size mismatch: expected={expected_bytes}, got={len(payload)}"
        )

    # Use a readonly view over payload for inputs to avoid unnecessary copies.
    array = np.frombuffer(payload, dtype=np_dtype)
    return array.reshape(normalized_shape)


def tensor_to_bytes(array: np.ndarray) -> bytes:
    if not isinstance(array, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(array)}")
    c_array = np.ascontiguousarray(array)
    return c_array.tobytes(order="C")


def parse_output_specs(items: Iterable[Mapping[str, object]]) -> List[TensorSpec]:
    specs: List[TensorSpec] = []
    for item in items:
        if not isinstance(item, Mapping):
            raise ValueError("Each output spec must be a JSON object")
        if "dtype" not in item or "shape" not in item:
            raise ValueError("Each output spec requires 'dtype' and 'shape'")
        name = str(item.get("name", f"output_{len(specs)}"))
        dtype = int(item["dtype"])
        raw_shape = item["shape"]
        if not isinstance(raw_shape, Sequence) or isinstance(raw_shape, (str, bytes)):
            raise ValueError(f"Output tensor '{name}' shape must be an array of dimensions")
        shape = normalize_shape(raw_shape)
        specs.append(TensorSpec(name=name, dtype=dtype, shape=shape))
    if not specs:
        raise ValueError("At least one output spec is required")
    return specs


def allocate_outputs(specs: Sequence[TensorSpec]) -> List[np.ndarray]:
    outputs: List[np.ndarray] = []
    for spec in specs:
        outputs.append(np.empty(spec.shape, dtype=dtype_code_to_numpy(spec.dtype), order="C"))
    return outputs


def tensors_to_json_payload(named_tensors: Sequence[Tuple[str, np.ndarray]]) -> List[MutableMapping[str, object]]:
    payload: List[MutableMapping[str, object]] = []
    for name, array in named_tensors:
        payload.append(
            {
                "name": name,
                "dtype": numpy_dtype_to_code(array.dtype),
                "shape": list(array.shape),
                "data_base64": base64.b64encode(tensor_to_bytes(array)).decode("ascii"),
            }
        )
    return payload


def tensors_from_json_payload(items: Sequence[Mapping[str, object]]) -> List[Tuple[str, np.ndarray]]:
    out: List[Tuple[str, np.ndarray]] = []
    for idx, item in enumerate(items):
        if not isinstance(item, Mapping):
            raise ValueError("Each input tensor entry must be a JSON object")
        if "dtype" not in item or "shape" not in item or "data_base64" not in item:
            raise ValueError("Each input tensor requires 'dtype', 'shape', and 'data_base64'")
        name = str(item.get("name", f"input_{idx}"))
        encoded = item["data_base64"]
        if not isinstance(encoded, str):
            raise ValueError(f"Input tensor '{name}' data_base64 must be a string")
        raw = base64.b64decode(encoded, validate=True)
        array = tensor_from_bytes(name, int(item["dtype"]), item["shape"], raw)
        out.append((name, array))
    if not out:
        raise ValueError("At least one input tensor is required")
    return out


def decode_npz_inputs(blob: bytes) -> List[Tuple[str, np.ndarray]]:
    stream = io.BytesIO(blob)
    try:
        with np.load(stream, allow_pickle=False) as archive:
            names = list(archive.files)
            if not names:
                raise ValueError("NPZ payload contains no tensors")
            tensors: List[Tuple[str, np.ndarray]] = []
            for name in names:
                arr = archive[name]
                if not isinstance(arr, np.ndarray):
                    raise ValueError(f"NPZ entry '{name}' is not an ndarray")
                tensors.append((name, arr))
            return _order_named_inputs(tensors)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Invalid NPZ payload: {exc}") from exc


def encode_npz_tensors(named_tensors: Sequence[Tuple[str, np.ndarray]]) -> bytes:
    stream = io.BytesIO()
    np.savez_compressed(stream, **{name: np.ascontiguousarray(array) for name, array in named_tensors})
    return stream.getvalue()


def _order_named_inputs(tensors: Sequence[Tuple[str, np.ndarray]]) -> List[Tuple[str, np.ndarray]]:
    """Preserve explicit numeric order for keys like input_0,input_1,... else lexical."""

    indexed: List[Tuple[int, str, np.ndarray]] = []
    fallback: List[Tuple[str, np.ndarray]] = []

    for name, array in tensors:
        if name.startswith("input_"):
            suffix = name[len("input_") :]
            if suffix.isdigit():
                indexed.append((int(suffix), name, array))
                continue
        fallback.append((name, array))

    if indexed and not fallback:
        indexed.sort(key=lambda x: x[0])
        return [(name, array) for _, name, array in indexed]

    # Mixed naming or custom names: keep deterministic lexical order.
    return sorted(list(tensors), key=lambda x: x[0])


def order_named_tensors(
    tensors: Sequence[Tuple[str, np.ndarray]], names_in_order: Sequence[str]
) -> List[Tuple[str, np.ndarray]]:
    by_name = {name: arr for name, arr in tensors}
    ordered: List[Tuple[str, np.ndarray]] = []
    for name in names_in_order:
        if name not in by_name:
            raise ValueError(f"Input tensor not found in payload: {name}")
        ordered.append((name, by_name[name]))
    if len(ordered) != len(tensors):
        extra = sorted([name for name, _ in tensors if name not in set(names_in_order)])
        if extra:
            raise ValueError(f"Input payload contains tensors not listed in requested order: {extra}")
    return ordered


def parse_json_header(
    value: str, field_name: str
) -> Union[Mapping[str, object], List[Mapping[str, object]]]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {field_name}: {exc}") from exc

    if not isinstance(parsed, (dict, list)):
        raise ValueError(f"{field_name} must be a JSON object or array")
    return parsed
