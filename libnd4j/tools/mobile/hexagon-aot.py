#!/usr/bin/env python3
"""Create and validate deterministic Qualcomm Hexagon/HTP SDX AOT requests."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import sys
import tempfile
from typing import Any, Dict, Iterable, List, Optional, Tuple

FORMAT_VERSION = 1
CACHE_ABI = "sdx-hexagon-aot-v1"
ADAPTER_ABI = 1
RANGE_SEMANTICS = "inclusive"
SOC_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")
ARTIFACT_PATTERN = re.compile(
    r"^hexagon_([0-9]+)_([0-9]+)_([0-9a-f]{16})[.](bin|meta)$"
)


class AotError(RuntimeError):
    pass


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def load_json(path: pathlib.Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AotError(f"cannot read JSON {path}: {exc}") from exc


def write_json(path: pathlib.Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def parse_shape_key(value: Any) -> int:
    try:
        parsed = int(str(value), 0)
    except (TypeError, ValueError) as exc:
        raise AotError(f"invalid shape key: {value!r}") from exc
    return parsed & ((1 << 64) - 1)


def artifact_base(start_slot: int, end_slot: int, shape_key: int) -> str:
    return f"hexagon_{start_slot}_{end_slot}_{shape_key:016x}"


def normalize_segment(
    segment: Dict[str, Any],
    include_noncapturable: bool,
    allow_unstable: bool,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    try:
        index = int(segment["index"])
        start_slot = int(segment["startSlot"])
        end_slot = int(segment["endSlot"])
        num_ops = int(segment["numOps"])
    except (KeyError, TypeError, ValueError) as exc:
        raise AotError(f"malformed segment entry: {segment!r}") from exc

    if start_slot < 0 or end_slot < start_slot:
        raise AotError(f"invalid inclusive segment range: {start_slot}..{end_slot}")
    expected_ops = end_slot - start_slot + 1
    if num_ops != expected_ops:
        raise AotError(
            f"segment {index} numOps={num_ops}, expected {expected_ops} "
            "for inclusive bounds"
        )

    shape_key = parse_shape_key(segment.get("shapeKey", 0))
    shape_status = str(segment.get("shapeKeyStatus", "UNSET"))
    reasons: List[str] = []
    if not bool(segment.get("isCapturable", False)) and not include_noncapturable:
        reasons.append("not-capturable")
    if bool(segment.get("compilationFailed", False)):
        reasons.append("prior-compilation-failed")
    if shape_key == 0:
        reasons.append("shape-key-unset")
    if shape_status != "STABLE" and not allow_unstable:
        reasons.append(f"shape-key-{shape_status.lower()}")

    base = artifact_base(start_slot, end_slot, shape_key)
    normalized: Dict[str, Any] = {
        "index": index,
        "startSlot": start_slot,
        "endSlot": end_slot,
        "numOps": num_ops,
        "rangeSemantics": RANGE_SEMANTICS,
        "shapeKey": shape_key,
        "shapeKeyHex": f"{shape_key:016x}",
        "shapeKeyStatus": shape_status,
        "artifact": f"{base}.bin",
        "metadata": f"{base}.meta",
        "ops": dict(sorted(dict(segment.get("ops", {})).items())),
    }
    if reasons:
        return None, {"index": index, "reasons": reasons}
    return normalized, None


def create_request(args: argparse.Namespace) -> Dict[str, Any]:
    source_path = pathlib.Path(args.segments_json)
    source = load_json(source_path)
    if not isinstance(source, list):
        raise AotError("segments JSON must be an array")

    if not SOC_PATTERN.fullmatch(args.soc):
        raise AotError(
            "SoC must contain only ASCII letters, digits, dot, underscore, or dash"
        )

    segments: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    for entry in source:
        if not isinstance(entry, dict):
            raise AotError(f"segment entry is not an object: {entry!r}")
        normalized, skip = normalize_segment(
            entry, args.include_noncapturable, args.allow_unstable
        )
        if normalized is not None:
            segments.append(normalized)
        if skip is not None:
            skipped.append(skip)

    segments.sort(key=lambda item: (item["startSlot"], item["endSlot"], item["index"]))
    artifact_names = [item["artifact"] for item in segments]
    if len(artifact_names) != len(set(artifact_names)):
        raise AotError("duplicate range/shape artifact names in replay plan")
    if not segments and not args.allow_empty:
        raise AotError("replay plan produced no eligible Hexagon AOT segments")

    request: Dict[str, Any] = {
        "formatVersion": FORMAT_VERSION,
        "cacheAbi": CACHE_ABI,
        "adapterAbi": ADAPTER_ABI,
        "soc": args.soc,
        "modelId": args.model_id or source_path.stem,
        "rangeSemantics": RANGE_SEMANTICS,
        "sourceSegmentsSha256": hashlib.sha256(
            canonical_json_bytes(source)
        ).hexdigest(),
        "segments": segments,
        "skippedSegments": skipped,
    }
    write_json(pathlib.Path(args.output), request)
    return request


def read_metadata(path: pathlib.Path) -> Dict[str, str]:
    result: Dict[str, str] = {}
    try:
        for line_number, raw_line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                raise AotError(f"{path}:{line_number}: expected key=value")
            key, value = line.split("=", 1)
            if key in result:
                raise AotError(f"{path}:{line_number}: duplicate key {key}")
            result[key] = value
    except OSError as exc:
        raise AotError(f"cannot read metadata {path}: {exc}") from exc
    return result


def expected_metadata(
    request: Dict[str, Any], segment: Dict[str, Any], payload: bytes
) -> Dict[str, str]:
    return {
        "cacheAbi": CACHE_ABI,
        "adapterAbi": str(ADAPTER_ABI),
        "soc": str(request["soc"]),
        "rangeSemantics": RANGE_SEMANTICS,
        "startSlot": str(segment["startSlot"]),
        "endSlot": str(segment["endSlot"]),
        "shapeKey": str(segment["shapeKeyHex"]),
        "byteSize": str(len(payload)),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def metadata_text(values: Dict[str, str]) -> str:
    order = (
        "cacheAbi",
        "adapterAbi",
        "soc",
        "rangeSemantics",
        "startSlot",
        "endSlot",
        "shapeKey",
        "byteSize",
        "sha256",
    )
    return "".join(f"{key}={values[key]}\n" for key in order)


def validate_request(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise AotError("AOT request must be an object")
    if value.get("formatVersion") != FORMAT_VERSION:
        raise AotError("unsupported AOT request formatVersion")
    if value.get("cacheAbi") != CACHE_ABI:
        raise AotError("unsupported Hexagon cache ABI")
    if value.get("adapterAbi") != ADAPTER_ABI:
        raise AotError("unsupported Hexagon adapter ABI")
    if value.get("rangeSemantics") != RANGE_SEMANTICS:
        raise AotError("Hexagon segment ranges must be inclusive")
    soc = str(value.get("soc", ""))
    if not SOC_PATTERN.fullmatch(soc):
        raise AotError("invalid or missing request SoC")
    segments = value.get("segments")
    if not isinstance(segments, list):
        raise AotError("request segments must be an array")
    return value


def verify_artifacts(
    request: Dict[str, Any], kernel_dir: pathlib.Path
) -> List[Dict[str, Any]]:
    if not kernel_dir.is_dir():
        raise AotError(f"kernel directory not found: {kernel_dir}")

    expected_bin_names = {str(segment["artifact"]) for segment in request["segments"]}
    actual_bin_names = {path.name for path in kernel_dir.glob("hexagon_*_*_*.bin")}
    unexpected = sorted(actual_bin_names - expected_bin_names)
    missing = sorted(expected_bin_names - actual_bin_names)
    if unexpected:
        raise AotError(f"unexpected Hexagon kernels: {', '.join(unexpected)}")
    if missing:
        raise AotError(f"missing Hexagon kernels: {', '.join(missing)}")

    manifest_entries: List[Dict[str, Any]] = []
    for segment in request["segments"]:
        bin_name = str(segment["artifact"])
        meta_name = str(segment["metadata"])
        if not ARTIFACT_PATTERN.fullmatch(bin_name):
            raise AotError(f"invalid kernel artifact name: {bin_name}")
        payload_path = kernel_dir / bin_name
        metadata_path = kernel_dir / meta_name
        payload = payload_path.read_bytes()
        if not payload:
            raise AotError(f"empty Hexagon kernel: {payload_path}")
        expected = expected_metadata(request, segment, payload)
        actual = read_metadata(metadata_path)
        if actual != expected:
            raise AotError(
                f"metadata mismatch for {bin_name}: expected {expected}, got {actual}"
            )
        manifest_entries.append(
            {
                "artifact": bin_name,
                "metadata": meta_name,
                "startSlot": segment["startSlot"],
                "endSlot": segment["endSlot"],
                "shapeKeyHex": segment["shapeKeyHex"],
                "byteSize": len(payload),
                "sha256": expected["sha256"],
            }
        )
    return manifest_entries


def finalize_artifacts(args: argparse.Namespace) -> Dict[str, Any]:
    request_path = pathlib.Path(args.request)
    request = validate_request(load_json(request_path))
    kernel_dir = pathlib.Path(args.kernel_dir)
    kernel_dir.mkdir(parents=True, exist_ok=True)

    for segment in request["segments"]:
        payload_path = kernel_dir / str(segment["artifact"])
        if not payload_path.is_file():
            raise AotError(f"compiled kernel is missing: {payload_path}")
        payload = payload_path.read_bytes()
        if not payload:
            raise AotError(f"compiled kernel is empty: {payload_path}")
        values = expected_metadata(request, segment, payload)
        metadata_path = kernel_dir / str(segment["metadata"])
        temporary = metadata_path.with_name(metadata_path.name + ".tmp")
        temporary.write_text(metadata_text(values), encoding="utf-8")
        temporary.replace(metadata_path)

    entries = verify_artifacts(request, kernel_dir)
    manifest = {
        "formatVersion": FORMAT_VERSION,
        "cacheAbi": CACHE_ABI,
        "adapterAbi": ADAPTER_ABI,
        "soc": request["soc"],
        "modelId": request["modelId"],
        "rangeSemantics": RANGE_SEMANTICS,
        "requestSha256": hashlib.sha256(request_path.read_bytes()).hexdigest(),
        "artifacts": entries,
    }
    write_json(kernel_dir / "hexagon-aot-manifest.json", manifest)
    return manifest


def verify_command(args: argparse.Namespace) -> Dict[str, Any]:
    request = validate_request(load_json(pathlib.Path(args.request)))
    entries = verify_artifacts(request, pathlib.Path(args.kernel_dir))
    return {"artifacts": entries}


def self_test() -> None:
    segments = [
        {
            "index": 0,
            "startSlot": 2,
            "endSlot": 3,
            "numOps": 2,
            "executionCount": 4,
            "isCapturable": True,
            "compilationFailed": False,
            "shapeKey": -1,
            "shapeKeyStatus": "STABLE",
            "ops": {"add": 1, "matmul": 1},
        },
        {
            "index": 1,
            "startSlot": 4,
            "endSlot": 4,
            "numOps": 1,
            "isCapturable": False,
            "compilationFailed": False,
            "shapeKey": 7,
            "shapeKeyStatus": "STABLE",
            "ops": {"identity": 1},
        },
    ]
    with tempfile.TemporaryDirectory(prefix="sdx-hexagon-aot-") as directory:
        root = pathlib.Path(directory)
        segments_path = root / "segments.json"
        request_path = root / "request.json"
        kernels = root / "kernels"
        write_json(segments_path, segments)
        args = argparse.Namespace(
            segments_json=str(segments_path),
            soc="SM8650",
            model_id="self-test",
            output=str(request_path),
            include_noncapturable=False,
            allow_unstable=False,
            allow_empty=False,
        )
        request = create_request(args)
        assert len(request["segments"]) == 1
        kernels.mkdir()
        artifact = request["segments"][0]["artifact"]
        (kernels / artifact).write_bytes(b"vendor-aot-test")
        finalize_artifacts(
            argparse.Namespace(request=str(request_path), kernel_dir=str(kernels))
        )
        verify_command(
            argparse.Namespace(request=str(request_path), kernel_dir=str(kernels))
        )
        (kernels / artifact).write_bytes(b"tampered")
        try:
            verify_command(
                argparse.Namespace(request=str(request_path), kernel_dir=str(kernels))
            )
        except AotError:
            pass
        else:
            raise AotError("self-test failed to reject a tampered kernel")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Turn functional replay segment JSON into deterministic Hexagon AOT "
            "compile requests and integrity-checked bundle artifacts."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="create a vendor compile request")
    plan.add_argument("--segments-json", required=True)
    plan.add_argument("--soc", required=True)
    plan.add_argument("--model-id")
    plan.add_argument("--output", required=True)
    plan.add_argument("--include-noncapturable", action="store_true")
    plan.add_argument("--allow-unstable", action="store_true")
    plan.add_argument("--allow-empty", action="store_true")

    finalize = subparsers.add_parser(
        "finalize", help="write sidecars and an integrity manifest for compiled bins"
    )
    finalize.add_argument("--request", required=True)
    finalize.add_argument("--kernel-dir", required=True)

    verify = subparsers.add_parser(
        "verify", help="verify compiled bins and metadata against a request"
    )
    verify.add_argument("--request", required=True)
    verify.add_argument("--kernel-dir", required=True)

    subparsers.add_parser("self-test", help="run the deterministic contract test")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "plan":
            request = create_request(args)
            print(
                f"wrote {len(request['segments'])} Hexagon AOT requests to "
                f"{args.output}"
            )
        elif args.command == "finalize":
            manifest = finalize_artifacts(args)
            print(
                f"finalized {len(manifest['artifacts'])} Hexagon AOT artifacts "
                f"under {args.kernel_dir}"
            )
        elif args.command == "verify":
            result = verify_command(args)
            print(f"verified {len(result['artifacts'])} Hexagon AOT artifacts")
        elif args.command == "self-test":
            self_test()
            print("Hexagon AOT contract self-test passed")
        else:
            raise AotError(f"unknown command: {args.command}")
    except (AotError, OSError, KeyError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
