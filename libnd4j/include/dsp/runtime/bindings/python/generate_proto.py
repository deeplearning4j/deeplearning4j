#!/usr/bin/env python3
"""Generate Python protobuf and gRPC stubs for SDX serving."""

from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate sdx_serving_pb2.py and sdx_serving_pb2_grpc.py")
    parser.add_argument(
        "--proto",
        default="sdx_serving.proto",
        help="Path to sdx_serving.proto (default: %(default)s in this directory)",
    )
    parser.add_argument(
        "--out",
        default=".",
        help="Output directory for generated modules (default: current directory)",
    )
    args = parser.parse_args()

    root = pathlib.Path(__file__).resolve().parent
    proto_path = pathlib.Path(args.proto)
    if not proto_path.is_absolute():
        proto_path = root / proto_path

    out_dir = pathlib.Path(args.out)
    if not out_dir.is_absolute():
        out_dir = (root / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not proto_path.exists():
        raise FileNotFoundError(f"Proto file not found: {proto_path}")

    cmd = [
        sys.executable,
        "-m",
        "grpc_tools.protoc",
        f"-I{proto_path.parent}",
        f"--python_out={out_dir}",
        f"--grpc_python_out={out_dir}",
        str(proto_path),
    ]
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            "Failed to generate protobuf stubs. Install dependencies with: "
            "pip install grpcio grpcio-tools"
        )

    print(f"Generated stubs in: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
