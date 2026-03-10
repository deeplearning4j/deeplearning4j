#!/usr/bin/env python3
"""SDX SDK runner exposing REST and gRPC inference endpoints.

This runner reuses the SDX C runtime Python wrapper and supports direct SDZ/SDNB
model loading through the existing `sdxLoadBundle(...)` runtime path.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import re
import sys
import threading
import time
import uuid
from concurrent import futures
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union
from urllib.parse import urlparse

from sdx_runtime import ModelOptions, RunOptions, SdxRuntime
from sdx_tensor_transport import (
    TensorSpec,
    allocate_outputs,
    decode_npz_inputs,
    encode_npz_tensors,
    numpy_dtype_to_code,
    order_named_tensors,
    parse_json_header,
    parse_output_specs,
    tensor_from_bytes,
    tensor_to_bytes,
    tensors_from_json_payload,
    tensors_to_json_payload,
)


def _to_execution_report_dict(report) -> Dict[str, Union[int, bool]]:
    return {
        "requested_backend": int(report.requested_backend),
        "applied_backend": int(report.applied_backend),
        "status_code": int(report.status_code),
        "used_fallback": bool(report.used_fallback),
        "execution_time_ns": int(report.execution_time_ns),
        "requested_gpu_target": int(report.requested_gpu_target),
        "applied_gpu_target": int(report.applied_gpu_target),
    }


def _parse_model_options(raw: Optional[Mapping[str, object]]) -> ModelOptions:
    data = raw or {}
    return ModelOptions(
        backend=int(data.get("backend", 0)),
        strict_backend=bool(data.get("strict_backend", False)),
        allow_runtime_jit=bool(data.get("allow_runtime_jit", False)),
        gpu_target=int(data.get("gpu_target", 0)),
    )


def _parse_run_options(raw: Optional[Mapping[str, object]]) -> RunOptions:
    data = raw or {}
    return RunOptions(
        backend=int(data.get("backend", 0)),
        strict_signature=bool(data.get("strict_signature", True)),
        gpu_target=int(data.get("gpu_target", 0)),
    )


@dataclass
class _ManagedModel:
    model: object
    context: object
    lock: threading.Lock


class SdxModelRegistry:
    """Thread-safe model registry over a single SDX runtime instance."""

    def __init__(
        self,
        library: Optional[str] = None,
        backend_preference: Optional[str] = None,
        platform_id: Optional[str] = None,
        library_dirs: Optional[Sequence[str]] = None,
        sdk_home: Optional[str] = None,
    ) -> None:
        self._runtime = SdxRuntime(
            library=library,
            backend_preference=backend_preference,
            platform_id=platform_id,
            library_dirs=library_dirs,
            sdk_home=sdk_home,
        )
        self._models: Dict[str, _ManagedModel] = {}
        self._lock = threading.Lock()

    def abi_version(self) -> int:
        return self._runtime.abi_version()

    def close(self) -> None:
        with self._lock:
            ids = list(self._models.keys())
        for model_id in ids:
            self.unload_model(model_id)
        self._runtime.close()

    def load_model(
        self,
        model_path: str,
        model_options: Optional[ModelOptions] = None,
        requested_outputs: Optional[Sequence[str]] = None,
    ) -> str:
        model = self._runtime.load_model(model_path, model_options)
        try:
            context = model.create_context(requested_outputs)
        except Exception:
            model.close()
            raise

        model_id = uuid.uuid4().hex
        with self._lock:
            self._models[model_id] = _ManagedModel(model=model, context=context, lock=threading.Lock())
        return model_id

    def unload_model(self, model_id: str) -> bool:
        with self._lock:
            managed = self._models.pop(model_id, None)

        if managed is None:
            return False

        try:
            managed.context.close()
        finally:
            managed.model.close()
        return True

    def run(
        self,
        model_id: str,
        inputs: Sequence[Tuple[str, object]],
        output_specs: Sequence[TensorSpec],
        run_options: Optional[RunOptions] = None,
    ) -> Tuple[List[Tuple[str, object]], Dict[str, Union[int, bool]]]:
        with self._lock:
            managed = self._models.get(model_id)
        if managed is None:
            raise KeyError(model_id)

        input_arrays = [arr for _, arr in inputs]
        output_arrays = allocate_outputs(output_specs)

        with managed.lock:
            managed.context.run(input_arrays, output_arrays, run_options)
            report = managed.context.execution_report()

        named_outputs = [(spec.name, out) for spec, out in zip(output_specs, output_arrays)]
        return named_outputs, _to_execution_report_dict(report)


class _SdxRestHandler(BaseHTTPRequestHandler):
    registry: SdxModelRegistry

    _LOAD_RE = re.compile(r"^/v1/models:load$")
    _UNLOAD_RE = re.compile(r"^/v1/models/([A-Za-z0-9_-]+):unload$")
    _RUN_JSON_RE = re.compile(r"^/v1/models/([A-Za-z0-9_-]+):run$")
    _RUN_NPZ_RE = re.compile(r"^/v1/models/([A-Za-z0-9_-]+):run-npz$")

    server_version = "SdxSdkRunner/1.0"

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/healthz":
            self._send_json(
                HTTPStatus.OK,
                {
                    "status": "ok",
                    "abi_version": self.registry.abi_version(),
                },
            )
            return

        self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path

        load_match = self._LOAD_RE.match(path)
        if load_match:
            self._handle_load_model()
            return

        unload_match = self._UNLOAD_RE.match(path)
        if unload_match:
            self._handle_unload_model(unload_match.group(1))
            return

        run_json_match = self._RUN_JSON_RE.match(path)
        if run_json_match:
            self._handle_run_json(run_json_match.group(1))
            return

        run_npz_match = self._RUN_NPZ_RE.match(path)
        if run_npz_match:
            self._handle_run_npz(run_npz_match.group(1))
            return

        self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})

    def log_message(self, fmt: str, *args) -> None:
        logging.info("REST %s - %s", self.client_address[0], fmt % args)

    def _read_body(self) -> bytes:
        length_raw = self.headers.get("Content-Length")
        if not length_raw:
            return b""
        try:
            length = int(length_raw)
        except ValueError as exc:
            raise ValueError("Invalid Content-Length") from exc
        if length < 0:
            raise ValueError("Invalid Content-Length")
        return self.rfile.read(length)

    def _read_json(self) -> Mapping[str, object]:
        body = self._read_body()
        try:
            payload = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON payload: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError("JSON payload must be an object")
        return payload

    def _send_json(
        self,
        status: HTTPStatus,
        payload: Mapping[str, object],
        extra_headers: Optional[Mapping[str, str]] = None,
    ) -> None:
        blob = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(blob)))
        if extra_headers:
            for key, value in extra_headers.items():
                self.send_header(key, value)
        self.end_headers()
        self.wfile.write(blob)

    def _send_binary(
        self,
        status: HTTPStatus,
        body: bytes,
        content_type: str,
        extra_headers: Optional[Mapping[str, str]] = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        if extra_headers:
            for key, value in extra_headers.items():
                self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _handle_load_model(self) -> None:
        try:
            payload = self._read_json()
            model_path = str(payload.get("model_path", "")).strip()
            if not model_path:
                raise ValueError("'model_path' is required")

            raw_model_options = payload.get("model_options")
            if raw_model_options is not None and not isinstance(raw_model_options, dict):
                raise ValueError("'model_options' must be an object")

            requested_outputs = payload.get("requested_outputs")
            if requested_outputs is not None:
                if not isinstance(requested_outputs, list):
                    raise ValueError("'requested_outputs' must be an array of strings")
                requested_outputs = [str(x) for x in requested_outputs]

            model_id = self.registry.load_model(
                model_path=model_path,
                model_options=_parse_model_options(raw_model_options),
                requested_outputs=requested_outputs,
            )
            self._send_json(
                HTTPStatus.OK,
                {
                    "model_id": model_id,
                    "abi_version": self.registry.abi_version(),
                },
            )
        except ValueError as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except Exception as exc:  # pragma: no cover - runtime errors
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})

    def _handle_unload_model(self, model_id: str) -> None:
        unloaded = self.registry.unload_model(model_id)
        if not unloaded:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Unknown model_id: {model_id}"})
            return
        self._send_json(HTTPStatus.OK, {"status": "unloaded", "model_id": model_id})

    def _handle_run_json(self, model_id: str) -> None:
        try:
            payload = self._read_json()

            raw_inputs = payload.get("inputs")
            if not isinstance(raw_inputs, list):
                raise ValueError("'inputs' must be an array")
            inputs = tensors_from_json_payload(raw_inputs)

            raw_outputs = payload.get("outputs")
            if not isinstance(raw_outputs, list):
                raise ValueError("'outputs' must be an array")
            output_specs = parse_output_specs(raw_outputs)

            raw_run_options = payload.get("run_options")
            if raw_run_options is not None and not isinstance(raw_run_options, dict):
                raise ValueError("'run_options' must be an object")

            outputs, report = self.registry.run(
                model_id=model_id,
                inputs=inputs,
                output_specs=output_specs,
                run_options=_parse_run_options(raw_run_options),
            )

            self._send_json(
                HTTPStatus.OK,
                {
                    "outputs": tensors_to_json_payload(outputs),
                    "report": report,
                },
            )
        except ValueError as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except KeyError:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Unknown model_id: {model_id}"})
        except Exception as exc:  # pragma: no cover - runtime errors
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})

    def _handle_run_npz(self, model_id: str) -> None:
        try:
            raw_specs_header = self.headers.get("X-SDX-Output-Specs")
            if not raw_specs_header:
                raise ValueError("Missing required header: X-SDX-Output-Specs")

            parsed_specs = parse_json_header(raw_specs_header, "X-SDX-Output-Specs")
            if not isinstance(parsed_specs, list):
                raise ValueError("X-SDX-Output-Specs must be a JSON array")
            output_specs = parse_output_specs(parsed_specs)

            raw_run_options_header = self.headers.get("X-SDX-Run-Options")
            run_options = None
            if raw_run_options_header:
                parsed_run_options = parse_json_header(raw_run_options_header, "X-SDX-Run-Options")
                if not isinstance(parsed_run_options, dict):
                    raise ValueError("X-SDX-Run-Options must be a JSON object")
                run_options = _parse_run_options(parsed_run_options)

            body = self._read_body()
            inputs = decode_npz_inputs(body)
            raw_input_order_header = self.headers.get("X-SDX-Input-Order")
            if raw_input_order_header:
                parsed_input_order = parse_json_header(raw_input_order_header, "X-SDX-Input-Order")
                if not isinstance(parsed_input_order, list) or not all(
                    isinstance(name, str) for name in parsed_input_order
                ):
                    raise ValueError("X-SDX-Input-Order must be a JSON string array")
                inputs = order_named_tensors(inputs, parsed_input_order)

            outputs, report = self.registry.run(
                model_id=model_id,
                inputs=inputs,
                output_specs=output_specs,
                run_options=run_options,
            )

            report_json = json.dumps(report, separators=(",", ":"))
            payload = encode_npz_tensors(outputs)
            self._send_binary(
                HTTPStatus.OK,
                payload,
                content_type="application/x-sdx-npz",
                extra_headers={"X-SDX-Execution-Report": report_json},
            )
        except ValueError as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except KeyError:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Unknown model_id: {model_id}"})
        except Exception as exc:  # pragma: no cover - runtime errors
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})


def _start_rest_server(registry: SdxModelRegistry, address: str, port: int) -> ThreadingHTTPServer:
    handler = _SdxRestHandler
    handler.registry = registry
    server = ThreadingHTTPServer((address, port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True, name="sdx-rest-server")
    thread.start()
    logging.info("REST server listening on http://%s:%d", address, port)
    return server


def _import_grpc_modules() -> Tuple[object, object, object]:
    try:
        import grpc  # type: ignore
    except ImportError as exc:  # pragma: no cover - env dependent
        raise RuntimeError("gRPC server requested but 'grpcio' is not installed") from exc

    module_dir = Path(__file__).resolve().parent
    module_dir_str = str(module_dir)
    if module_dir_str not in sys.path:
        sys.path.insert(0, module_dir_str)

    try:
        pb2 = importlib.import_module("sdx_serving_pb2")
        pb2_grpc = importlib.import_module("sdx_serving_pb2_grpc")
        return grpc, pb2, pb2_grpc
    except ImportError:
        pass

    try:
        from grpc_tools import protoc  # type: ignore
    except ImportError as exc:  # pragma: no cover - env dependent
        raise RuntimeError(
            "gRPC stubs are not generated and grpcio-tools is unavailable. "
            "Run: python generate_proto.py"
        ) from exc

    proto_path = module_dir / "sdx_serving.proto"
    code = protoc.main(
        [
            "protoc",
            f"-I{module_dir}",
            f"--python_out={module_dir}",
            f"--grpc_python_out={module_dir}",
            str(proto_path),
        ]
    )
    if code != 0:
        raise RuntimeError(f"Failed to generate gRPC stubs from {proto_path}")

    importlib.invalidate_caches()
    pb2 = importlib.import_module("sdx_serving_pb2")
    pb2_grpc = importlib.import_module("sdx_serving_pb2_grpc")
    return grpc, pb2, pb2_grpc


def _start_grpc_server(
    registry: SdxModelRegistry,
    address: str,
    port: int,
    workers: int,
    max_message_bytes: int,
):
    grpc, pb2, pb2_grpc = _import_grpc_modules()

    class _GrpcService(pb2_grpc.SdxRuntimeServiceServicer):
        def Health(self, request, context):  # noqa: N802
            del request
            return pb2.HealthResponse(status="ok", abi_version=registry.abi_version())

        def LoadModel(self, request, context):  # noqa: N802
            try:
                model_options = None
                if request.HasField("options"):
                    model_options = ModelOptions(
                        backend=int(request.options.backend),
                        strict_backend=bool(request.options.strict_backend),
                        allow_runtime_jit=bool(request.options.allow_runtime_jit),
                        gpu_target=int(request.options.gpu_target),
                    )

                model_id = registry.load_model(
                    model_path=request.model_path,
                    model_options=model_options,
                    requested_outputs=list(request.requested_outputs),
                )
                return pb2.LoadModelResponse(model_id=model_id, abi_version=registry.abi_version())
            except Exception as exc:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details(str(exc))
                return pb2.LoadModelResponse()

        def UnloadModel(self, request, context):  # noqa: N802
            unloaded = registry.unload_model(request.model_id)
            if not unloaded:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details(f"Unknown model_id: {request.model_id}")
            return pb2.UnloadModelResponse()

        def Run(self, request, context):  # noqa: N802
            try:
                inputs: List[Tuple[str, object]] = []
                for idx, tensor in enumerate(request.inputs):
                    name = tensor.name or f"input_{idx}"
                    arr = tensor_from_bytes(name, int(tensor.dtype), tensor.shape, bytes(tensor.data))
                    inputs.append((name, arr))

                specs = parse_output_specs(
                    [
                        {
                            "name": spec.name or f"output_{idx}",
                            "dtype": int(spec.dtype),
                            "shape": list(spec.shape),
                        }
                        for idx, spec in enumerate(request.outputs)
                    ]
                )
                if not specs:
                    raise ValueError("At least one output spec is required")

                run_options = None
                if request.HasField("options"):
                    run_options = RunOptions(
                        backend=int(request.options.backend),
                        strict_signature=bool(request.options.strict_signature),
                        gpu_target=int(request.options.gpu_target),
                    )

                outputs, report = registry.run(
                    model_id=request.model_id,
                    inputs=inputs,
                    output_specs=specs,
                    run_options=run_options,
                )

                response = pb2.RunResponse()
                for name, arr in outputs:
                    response.outputs.add(
                        name=name,
                        dtype=numpy_dtype_to_code(arr.dtype),
                    )
                    response.outputs[-1].shape.extend(arr.shape)
                    response.outputs[-1].data = tensor_to_bytes(arr)

                response.report.requested_backend = int(report["requested_backend"])
                response.report.applied_backend = int(report["applied_backend"])
                response.report.status_code = int(report["status_code"])
                response.report.used_fallback = bool(report["used_fallback"])
                response.report.execution_time_ns = int(report["execution_time_ns"])
                response.report.requested_gpu_target = int(report["requested_gpu_target"])
                response.report.applied_gpu_target = int(report["applied_gpu_target"])
                return response
            except KeyError:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details(f"Unknown model_id: {request.model_id}")
                return pb2.RunResponse()
            except Exception as exc:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details(str(exc))
                return pb2.RunResponse()

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=workers),
        options=[
            ("grpc.max_send_message_length", max_message_bytes),
            ("grpc.max_receive_message_length", max_message_bytes),
        ],
    )
    pb2_grpc.add_SdxRuntimeServiceServicer_to_server(_GrpcService(), server)
    bind_addr = f"{address}:{port}"
    server.add_insecure_port(bind_addr)
    server.start()
    logging.info("gRPC server listening on %s", bind_addr)
    return server


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve SDX runtime over REST and gRPC")
    parser.add_argument("--library", default=None, help="Path/name of runtime library (libnd4jcpu/libnd4jcuda/libnd4jamd)")
    parser.add_argument("--runtime-backend", default=None, help="Preferred runtime backend lookup order seed (cpu|cuda|amd)")
    parser.add_argument("--runtime-platform-id", default=None, help="Override detected platform id (for packaged SDK lookup)")
    parser.add_argument("--runtime-home", default=None, help="Root directory of packaged SDX runtime SDK")
    parser.add_argument(
        "--runtime-lib-dir",
        action="append",
        default=[],
        help="Additional library search directory (repeatable)",
    )

    parser.add_argument("--rest-address", default="0.0.0.0", help="REST bind address")
    parser.add_argument("--rest-port", type=int, default=8080, help="REST bind port")
    parser.add_argument("--disable-rest", action="store_true", help="Disable REST server")

    parser.add_argument("--grpc-address", default="0.0.0.0", help="gRPC bind address")
    parser.add_argument("--grpc-port", type=int, default=50051, help="gRPC bind port")
    parser.add_argument("--grpc-workers", type=int, default=8, help="gRPC worker thread count")
    parser.add_argument(
        "--grpc-max-message-bytes",
        type=int,
        default=64 * 1024 * 1024,
        help="gRPC max send/receive message size",
    )
    parser.add_argument("--disable-grpc", action="store_true", help="Disable gRPC server")

    parser.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(message)s")

    if args.disable_rest and args.disable_grpc:
        raise SystemExit("At least one protocol must be enabled")

    registry = SdxModelRegistry(
        library=args.library,
        backend_preference=args.runtime_backend,
        platform_id=args.runtime_platform_id,
        library_dirs=args.runtime_lib_dir,
        sdk_home=args.runtime_home,
    )
    rest_server: Optional[ThreadingHTTPServer] = None
    grpc_server = None

    try:
        if not args.disable_rest:
            rest_server = _start_rest_server(registry, args.rest_address, args.rest_port)

        if not args.disable_grpc:
            grpc_server = _start_grpc_server(
                registry=registry,
                address=args.grpc_address,
                port=args.grpc_port,
                workers=args.grpc_workers,
                max_message_bytes=args.grpc_max_message_bytes,
            )

        if grpc_server is not None:
            grpc_server.wait_for_termination()
        else:
            while True:
                time.sleep(60)
    except KeyboardInterrupt:
        logging.info("Shutting down SDX SDK runner")
    finally:
        if grpc_server is not None:
            grpc_server.stop(grace=2)
        if rest_server is not None:
            rest_server.shutdown()
            rest_server.server_close()
        registry.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
