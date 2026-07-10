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
import sys
import threading
import uuid
from concurrent import futures
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from fastapi import FastAPI, Header, HTTPException, Request, Response
from pydantic import BaseModel, Field

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
        "plan_phase": int(getattr(report, "plan_phase", -1)),
        "execution_count": int(getattr(report, "execution_count", -1)),
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

        input_arrays = self._order_inputs_for_plan(managed, inputs)
        output_arrays = allocate_outputs(output_specs)

        with managed.lock:
            managed.context.run(input_arrays, output_arrays, run_options)
            report = managed.context.execution_report()

        named_outputs = [(spec.name, out) for spec, out in zip(output_specs, output_arrays)]
        return named_outputs, _to_execution_report_dict(report)

    @staticmethod
    def _order_inputs_for_plan(
        managed: "_ManagedModel", inputs: Sequence[Tuple[str, object]]
    ) -> List[object]:
        """Reorder named request tensors to the plan's external input order.

        The plan binds external inputs positionally (constants, variables, and
        placeholders). When the context exposes input names and every request
        tensor name matches, requests may arrive in any order; otherwise the
        client-provided order is trusted as-is.
        """
        by_name = {name: arr for name, arr in inputs}
        try:
            plan_names = managed.context.input_names()
        except Exception:
            plan_names = []
        if (
            plan_names
            and len(plan_names) == len(inputs)
            and len(by_name) == len(inputs)
            and all(name in by_name for name in plan_names)
        ):
            return [by_name[name] for name in plan_names]
        return [arr for _, arr in inputs]


# ---------------------------------------------------------------------------
# Pydantic request/response models
# ---------------------------------------------------------------------------


class LoadModelRequest(BaseModel):
    model_path: str
    model_options: Optional[Dict[str, Any]] = None
    requested_outputs: Optional[List[str]] = None


class LoadModelResponse(BaseModel):
    model_id: str
    abi_version: int


class UnloadModelResponse(BaseModel):
    status: str
    model_id: str


class RunJsonRequest(BaseModel):
    inputs: List[Any]
    outputs: List[Any]
    run_options: Optional[Dict[str, Any]] = None


class RunJsonResponse(BaseModel):
    outputs: List[Any]
    report: Dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    abi_version: int


# ---------------------------------------------------------------------------
# FastAPI application factory
# ---------------------------------------------------------------------------


def create_app(registry: SdxModelRegistry) -> FastAPI:
    """Create a FastAPI application wired to the given model registry."""
    app = FastAPI(title="SDX SDK Runner", version="1.0")

    @app.get("/healthz", response_model=HealthResponse)
    async def health():
        return {"status": "ok", "abi_version": registry.abi_version()}

    @app.post("/v1/models:load", response_model=LoadModelResponse)
    async def load_model(req: LoadModelRequest):
        model_path = req.model_path.strip()
        if not model_path:
            raise HTTPException(status_code=400, detail="'model_path' is required")

        model_id = registry.load_model(
            model_path=model_path,
            model_options=_parse_model_options(req.model_options),
            requested_outputs=req.requested_outputs,
        )
        return {"model_id": model_id, "abi_version": registry.abi_version()}

    @app.post("/v1/models/{model_id}:unload", response_model=UnloadModelResponse)
    async def unload_model(model_id: str):
        unloaded = registry.unload_model(model_id)
        if not unloaded:
            raise HTTPException(status_code=404, detail=f"Unknown model_id: {model_id}")
        return {"status": "unloaded", "model_id": model_id}

    @app.post("/v1/models/{model_id}:run", response_model=RunJsonResponse)
    async def run_json(model_id: str, req: RunJsonRequest):
        try:
            inputs = tensors_from_json_payload(req.inputs)
            output_specs = parse_output_specs(req.outputs)

            outputs, report = registry.run(
                model_id=model_id,
                inputs=inputs,
                output_specs=output_specs,
                run_options=_parse_run_options(req.run_options),
            )
            return {"outputs": tensors_to_json_payload(outputs), "report": report}
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Unknown model_id: {model_id}")

    @app.post("/v1/models/{model_id}:run-npz")
    async def run_npz(
        model_id: str,
        request: Request,
        x_sdx_output_specs: str = Header(..., alias="X-SDX-Output-Specs"),
        x_sdx_run_options: Optional[str] = Header(None, alias="X-SDX-Run-Options"),
        x_sdx_input_order: Optional[str] = Header(None, alias="X-SDX-Input-Order"),
    ):
        parsed_specs = parse_json_header(x_sdx_output_specs, "X-SDX-Output-Specs")
        if not isinstance(parsed_specs, list):
            raise HTTPException(status_code=400, detail="X-SDX-Output-Specs must be a JSON array")
        output_specs = parse_output_specs(parsed_specs)

        run_options = None
        if x_sdx_run_options:
            parsed_run_options = parse_json_header(x_sdx_run_options, "X-SDX-Run-Options")
            if not isinstance(parsed_run_options, dict):
                raise HTTPException(status_code=400, detail="X-SDX-Run-Options must be a JSON object")
            run_options = _parse_run_options(parsed_run_options)

        body = await request.body()
        inputs = decode_npz_inputs(body)

        if x_sdx_input_order:
            parsed_input_order = parse_json_header(x_sdx_input_order, "X-SDX-Input-Order")
            if not isinstance(parsed_input_order, list) or not all(
                isinstance(name, str) for name in parsed_input_order
            ):
                raise HTTPException(status_code=400, detail="X-SDX-Input-Order must be a JSON string array")
            inputs = order_named_tensors(inputs, parsed_input_order)

        try:
            outputs, report = registry.run(
                model_id=model_id,
                inputs=inputs,
                output_specs=output_specs,
                run_options=run_options,
            )
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Unknown model_id: {model_id}")

        report_json = json.dumps(report, separators=(",", ":"))
        payload = encode_npz_tensors(outputs)
        return Response(
            content=payload,
            media_type="application/x-sdx-npz",
            headers={"X-SDX-Execution-Report": report_json},
        )

    return app


# ---------------------------------------------------------------------------
# gRPC server (unchanged — uses grpcio directly)
# ---------------------------------------------------------------------------


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
                response.report.plan_phase = int(report.get("plan_phase", -1))
                response.report.execution_count = int(report.get("execution_count", -1))
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


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


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
    grpc_server = None

    try:
        if not args.disable_rest:
            import uvicorn  # type: ignore
            app = create_app(registry)
            config = uvicorn.Config(
                app,
                host=args.rest_address,
                port=args.rest_port,
                log_level=args.log_level.lower(),
            )
            server = uvicorn.Server(config)
            rest_thread = threading.Thread(target=server.run, daemon=True, name="sdx-rest-server")
            rest_thread.start()
            logging.info("REST server (FastAPI/Uvicorn) listening on http://%s:%d", args.rest_address, args.rest_port)

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
            import time
            while True:
                time.sleep(60)
    except KeyboardInterrupt:
        logging.info("Shutting down SDX SDK runner")
    finally:
        if grpc_server is not None:
            grpc_server.stop(grace=2)
        registry.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
