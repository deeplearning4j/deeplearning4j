"""Python ctypes wrapper for the SDX runtime C ABI."""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

try:
    import numpy as _np  # type: ignore
except Exception:  # pragma: no cover
    _np = None


SDX_STATUS_OK = 0

SDX_BACKEND_AUTO = 0
SDX_BACKEND_SLOT_BY_SLOT = 1
SDX_BACKEND_CUDA_GRAPHS = 2
SDX_BACKEND_NVRTC = 3
SDX_BACKEND_PTX = 4
SDX_BACKEND_TRITON = 5
SDX_BACKEND_MLX = 6
SDX_BACKEND_ARM_HYBRID = 7
SDX_BACKEND_NNAPI = 8

SDX_DEVICE_HOST = 0
SDX_DEVICE_CUDA = 1
SDX_DEVICE_AMD = 2

SDX_GPU_TARGET_AUTO = 0
SDX_GPU_TARGET_CUDA = 1
SDX_GPU_TARGET_AMD = 2


_SDX_BACKENDS: Tuple[str, ...] = ("cpu", "cuda", "amd")
_SDX_BACKEND_TO_LIB: Dict[str, str] = {
    "cpu": "nd4jcpu",
    "cuda": "nd4jcuda",
    "amd": "nd4jamd",
}


def _normalize_arch(machine: str) -> str:
    value = machine.strip().lower()
    mapping = {
        "x86_64": "x86_64",
        "amd64": "x86_64",
        "x64": "x86_64",
        "aarch64": "arm64",
        "arm64": "arm64",
        "armv8l": "arm64",
        "i386": "x86",
        "i686": "x86",
    }
    return mapping.get(value, value or "unknown")


def detect_host_platform_id(
    sys_platform: Optional[str] = None,
    machine: Optional[str] = None,
    os_name: Optional[str] = None,
    environ: Optional[Mapping[str, str]] = None,
) -> str:
    env = environ or os.environ
    sp = (sys_platform or sys.platform).strip().lower()
    osn = (os_name or os.name).strip().lower()
    arch = _normalize_arch(machine or platform.machine())

    if osn == "nt" or sp.startswith("win"):
        os_id = "windows"
    elif sp.startswith("ios"):
        os_id = "ios"
    elif sp == "darwin":
        os_id = "macos"
    elif "android" in sp or ("ANDROID_ROOT" in env and "ANDROID_DATA" in env):
        os_id = "android"
    elif sp.startswith("linux"):
        os_id = "linux"
    else:
        os_id = sp.split("-", 1)[0] if sp else "unknown"

    return f"{os_id}-{arch}"


def _shared_library_extension(platform_id: str) -> str:
    if platform_id.startswith("windows-"):
        return ".dll"
    if platform_id.startswith("macos-") or platform_id.startswith("ios-"):
        return ".dylib"
    return ".so"


def _runtime_library_filename(platform_id: str, backend: str) -> str:
    lib_base = _SDX_BACKEND_TO_LIB[backend]
    ext = _shared_library_extension(platform_id)
    if platform_id.startswith("windows-"):
        return f"{lib_base}{ext}"
    return f"lib{lib_base}{ext}"


def _backend_priority(preferred_backend: Optional[str]) -> List[str]:
    preferred = (preferred_backend or "").strip().lower()
    if not preferred:
        return list(_SDX_BACKENDS)
    if preferred not in _SDX_BACKENDS:
        raise ValueError(
            f"Invalid backend preference: {preferred_backend}. "
            f"Expected one of: {', '.join(_SDX_BACKENDS)}"
        )
    return [preferred] + [b for b in _SDX_BACKENDS if b != preferred]


def _split_env_paths(raw: str) -> List[Path]:
    out: List[Path] = []
    for part in raw.split(os.pathsep):
        value = part.strip()
        if value:
            out.append(Path(value).expanduser().resolve())
    return out


def _build_search_roots(module_dir: Path, environ: Mapping[str, str]) -> List[Path]:
    roots: List[Path] = [module_dir, module_dir.parent]
    env_home = environ.get("SDX_RUNTIME_HOME", "").strip()
    if env_home:
        roots.insert(0, Path(env_home).expanduser().resolve())
    return roots


def _collect_packaged_library_candidates(
    roots: Sequence[Path], platform_id: str, backends: Sequence[str], environ: Mapping[str, str]
) -> List[str]:
    ext = _shared_library_extension(platform_id)
    candidates: List[str] = []
    seen: Set[str] = set()

    dir_override = environ.get("SDX_RUNTIME_LIBRARY_DIR", "").strip()
    direct_dirs: List[Path] = _split_env_paths(dir_override) if dir_override else []

    for root in roots:
        direct_dirs.extend(
            [
                root / "lib",
                root / "sdx-runtime-sdk" / "lib",
            ]
        )
        for backend in backends:
            direct_dirs.extend(
                [
                    root / "bindings" / platform_id / backend / "lib",
                    root / "sdx-runtime-sdk" / "bindings" / platform_id / backend / "lib",
                    root / platform_id / backend / "lib",
                ]
            )

    for directory in direct_dirs:
        if not directory.exists() or not directory.is_dir():
            continue

        for backend in backends:
            base = _runtime_library_filename(platform_id, backend)
            patterns = [base]
            if ext in (".so", ".dylib"):
                patterns.append(base + ".*")

            for pattern in patterns:
                for candidate_path in sorted(directory.glob(pattern)):
                    candidate = str(candidate_path)
                    if candidate not in seen:
                        seen.add(candidate)
                        candidates.append(candidate)

    return candidates


def _system_library_names(platform_id: str, backends: Sequence[str]) -> List[str]:
    names: List[str] = []
    for backend in backends:
        lib_base = _SDX_BACKEND_TO_LIB[backend]
        runtime_filename = _runtime_library_filename(platform_id, backend)
        names.extend([runtime_filename, lib_base])
    return names


def _build_library_search_plan(
    explicit_library: Optional[str],
    platform_id: str,
    preferred_backend: Optional[str],
    module_dir: Path,
    environ: Mapping[str, str],
) -> List[str]:
    plan: List[str] = []
    seen: Set[str] = set()

    if explicit_library:
        plan.append(explicit_library)
        seen.add(explicit_library)

    backends = _backend_priority(preferred_backend)
    packaged = _collect_packaged_library_candidates(
        roots=_build_search_roots(module_dir, environ),
        platform_id=platform_id,
        backends=backends,
        environ=environ,
    )
    for candidate in packaged:
        if candidate not in seen:
            seen.add(candidate)
            plan.append(candidate)

    for candidate in _system_library_names(platform_id, backends):
        if candidate not in seen:
            seen.add(candidate)
            plan.append(candidate)

    return plan


def _dtype_code_from_numpy_dtype(dtype) -> int:
    if _np is None:
        raise RuntimeError("NumPy is not available")

    dt = _np.dtype(dtype)
    mapping = {
        _np.dtype(_np.bool_): 1,
        _np.dtype(_np.float16): 3,
        _np.dtype(_np.float32): 5,
        _np.dtype(_np.float64): 6,
        _np.dtype(_np.int8): 7,
        _np.dtype(_np.int16): 8,
        _np.dtype(_np.int32): 9,
        _np.dtype(_np.int64): 10,
        _np.dtype(_np.uint8): 11,
        _np.dtype(_np.uint16): 12,
        _np.dtype(_np.uint32): 13,
        _np.dtype(_np.uint64): 14,
    }
    if hasattr(_np, "bfloat16"):
        mapping[_np.dtype(_np.bfloat16)] = 17

    code = mapping.get(dt)
    if code is None:
        raise TypeError(f"Unsupported NumPy dtype for SDX runtime: {dt}")
    return code


class _NumpyTensorLease:
    def __init__(self, array, is_output: bool) -> None:
        if _np is None:
            raise RuntimeError("NumPy support requested but NumPy is not installed")
        if not isinstance(array, _np.ndarray):
            raise TypeError(f"Expected numpy.ndarray, got: {type(array)}")

        self._original = array
        self._working = array
        self._copy_back = False

        if is_output:
            if not array.flags.writeable:
                raise ValueError("Output NumPy arrays must be writeable")
            if not array.flags.c_contiguous:
                self._working = _np.empty(array.shape, dtype=array.dtype, order="C")
                self._copy_back = True
        else:
            if not array.flags.c_contiguous:
                self._working = _np.ascontiguousarray(array)

        self.view = TensorView.from_raw(
            int(self._working.__array_interface__["data"][0]),
            self._working.shape,
            _dtype_code_from_numpy_dtype(self._working.dtype),
            int(self._working.nbytes),
            SDX_DEVICE_HOST,
            -1,
        )

    def finalize(self) -> None:
        if self._copy_back:
            _np.copyto(self._original, self._working, casting="no")


class RuntimeOptions(ctypes.Structure):
    _fields_ = [("struct_size", ctypes.c_uint32)]

    def __init__(self) -> None:
        super().__init__(ctypes.sizeof(RuntimeOptions))


class ModelOptions(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("backend", ctypes.c_int32),
        ("strict_backend", ctypes.c_int32),
        ("allow_runtime_jit", ctypes.c_int32),
        ("gpu_target", ctypes.c_int32),
    ]

    def __init__(
        self,
        backend: int = SDX_BACKEND_AUTO,
        strict_backend: bool = False,
        allow_runtime_jit: bool = False,
        gpu_target: int = SDX_GPU_TARGET_AUTO,
    ) -> None:
        super().__init__(
            ctypes.sizeof(ModelOptions),
            backend,
            1 if strict_backend else 0,
            1 if allow_runtime_jit else 0,
            gpu_target,
        )


class RunOptions(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("backend", ctypes.c_int32),
        ("strict_signature", ctypes.c_int32),
        ("gpu_target", ctypes.c_int32),
    ]

    def __init__(
        self,
        backend: int = SDX_BACKEND_AUTO,
        strict_signature: bool = True,
        gpu_target: int = SDX_GPU_TARGET_AUTO,
    ) -> None:
        super().__init__(
            ctypes.sizeof(RunOptions),
            backend,
            1 if strict_signature else 0,
            gpu_target,
        )


class TensorView(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("rank", ctypes.c_int32),
        ("dtype", ctypes.c_int32),
        ("bytes", ctypes.c_size_t),
        ("device_type", ctypes.c_int32),
        ("device_id", ctypes.c_int32),
    ]

    @staticmethod
    def from_raw(
        data_ptr: int | ctypes.c_void_p,
        shape: Sequence[int],
        dtype: int,
        nbytes: int,
        device_type: int = SDX_DEVICE_HOST,
        device_id: int = -1,
    ) -> "TensorView":
        shape_arr = (ctypes.c_int64 * len(shape))(*shape)
        tv = TensorView()
        tv.data = ctypes.c_void_p(int(data_ptr)) if not isinstance(data_ptr, ctypes.c_void_p) else data_ptr
        tv.shape = ctypes.cast(shape_arr, ctypes.POINTER(ctypes.c_int64)) if shape else None
        tv.rank = len(shape)
        tv.dtype = int(dtype)
        tv.bytes = int(nbytes)
        tv.device_type = int(device_type)
        tv.device_id = int(device_id)
        tv._shape_owner = shape_arr
        return tv


class ExecutionReport(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("requested_backend", ctypes.c_int32),
        ("applied_backend", ctypes.c_int32),
        ("status_code", ctypes.c_int32),
        ("used_fallback", ctypes.c_int32),
        ("execution_time_ns", ctypes.c_uint64),
        ("requested_gpu_target", ctypes.c_int32),
        ("applied_gpu_target", ctypes.c_int32),
    ]

    def __init__(self) -> None:
        super().__init__(ctypes.sizeof(ExecutionReport))


@dataclass
class _Handles:
    runtime: ctypes.c_void_p
    model: Optional[ctypes.c_void_p] = None
    context: Optional[ctypes.c_void_p] = None


class SdxRuntime:
    def __init__(
        self,
        library: Optional[str] = None,
        backend_preference: Optional[str] = None,
        platform_id: Optional[str] = None,
        library_dirs: Optional[Sequence[str]] = None,
        sdk_home: Optional[str] = None,
    ) -> None:
        self._lib = self._load_library(
            library=library,
            backend_preference=backend_preference,
            platform_id=platform_id,
            library_dirs=library_dirs,
            sdk_home=sdk_home,
        )
        self._bind_prototypes()
        runtime = ctypes.c_void_p()
        status = self._lib.sdxCreateRuntime(None, ctypes.byref(runtime))
        self._raise_on_error(status, runtime, "sdxCreateRuntime")
        self._handles = _Handles(runtime=runtime)

    @staticmethod
    def _load_library(
        library: Optional[str],
        backend_preference: Optional[str] = None,
        platform_id: Optional[str] = None,
        library_dirs: Optional[Sequence[str]] = None,
        sdk_home: Optional[str] = None,
    ) -> ctypes.CDLL:
        loader_env: Dict[str, str] = dict(os.environ)
        if sdk_home:
            loader_env["SDX_RUNTIME_HOME"] = sdk_home
        if library_dirs:
            existing = loader_env.get("SDX_RUNTIME_LIBRARY_DIR", "").strip()
            provided = os.pathsep.join([str(Path(v).expanduser()) for v in library_dirs if str(v).strip()])
            if existing and provided:
                loader_env["SDX_RUNTIME_LIBRARY_DIR"] = provided + os.pathsep + existing
            elif provided:
                loader_env["SDX_RUNTIME_LIBRARY_DIR"] = provided

        resolved_platform_id = (platform_id or loader_env.get("SDX_RUNTIME_PLATFORM_ID", "")).strip()
        if not resolved_platform_id:
            resolved_platform_id = detect_host_platform_id(environ=loader_env)
        preferred_backend = (backend_preference or loader_env.get("SDX_RUNTIME_BACKEND", "")).strip() or None
        module_dir = Path(__file__).resolve().parent

        candidates = _build_library_search_plan(
            explicit_library=library,
            platform_id=resolved_platform_id,
            preferred_backend=preferred_backend,
            module_dir=module_dir,
            environ=loader_env,
        )

        errors: List[str] = []
        for name in candidates:
            has_path_sep = os.path.sep in name or (os.path.altsep and os.path.altsep in name)
            has_ext = name.endswith((".dll", ".so", ".dylib"))
            path = None
            if not has_path_sep and not has_ext:
                path = ctypes.util.find_library(name)
            try_name = path or name
            try:
                return ctypes.CDLL(try_name)
            except OSError as exc:
                errors.append(f"{try_name}: {exc}")
                continue

        debug_hint = (
            "Unable to load SDX runtime library. "
            "Set explicit path via SdxRuntime(library=...), or configure "
            "SDX_RUNTIME_HOME / SDX_RUNTIME_LIBRARY_DIR / SDX_RUNTIME_BACKEND. "
            f"Detected platform={resolved_platform_id}. Tried: {', '.join(candidates)}"
        )
        if errors:
            raise OSError(debug_hint + f". Last error: {errors[-1]}")
        raise OSError(debug_hint)

    def _bind_prototypes(self) -> None:
        self._lib.sdxGetRuntimeAbiVersion.argtypes = []
        self._lib.sdxGetRuntimeAbiVersion.restype = ctypes.c_int

        self._lib.sdxCreateRuntime.argtypes = [ctypes.POINTER(RuntimeOptions), ctypes.POINTER(ctypes.c_void_p)]
        self._lib.sdxCreateRuntime.restype = ctypes.c_int

        self._lib.sdxDestroyRuntime.argtypes = [ctypes.c_void_p]
        self._lib.sdxDestroyRuntime.restype = None

        self._lib.sdxLoadBundle.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.POINTER(ModelOptions),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self._lib.sdxLoadBundle.restype = ctypes.c_int

        self._lib.sdxUnloadModel.argtypes = [ctypes.c_void_p]
        self._lib.sdxUnloadModel.restype = None

        self._lib.sdxCreateContext.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_char_p),
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self._lib.sdxCreateContext.restype = ctypes.c_int

        self._lib.sdxDestroyContext.argtypes = [ctypes.c_void_p]
        self._lib.sdxDestroyContext.restype = None

        self._lib.sdxRun.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(TensorView),
            ctypes.c_int32,
            ctypes.POINTER(TensorView),
            ctypes.c_int32,
            ctypes.POINTER(RunOptions),
        ]
        self._lib.sdxRun.restype = ctypes.c_int

        self._lib.sdxGetLastError.argtypes = [ctypes.c_void_p]
        self._lib.sdxGetLastError.restype = ctypes.c_char_p

        self._lib.sdxGetExecutionReport.argtypes = [ctypes.c_void_p, ctypes.POINTER(ExecutionReport)]
        self._lib.sdxGetExecutionReport.restype = ctypes.c_int

    def abi_version(self) -> int:
        return int(self._lib.sdxGetRuntimeAbiVersion())

    def load_model(self, bundle_path: str, options: Optional[ModelOptions] = None) -> "SdxModel":
        if options is None:
            options_ptr = None
        else:
            options_ptr = ctypes.byref(options)

        model = ctypes.c_void_p()
        status = self._lib.sdxLoadBundle(
            self._handles.runtime,
            bundle_path.encode("utf-8"),
            options_ptr,
            ctypes.byref(model),
        )
        self._raise_on_error(status, self._handles.runtime, "sdxLoadBundle")
        return SdxModel(self, model)

    def last_error(self) -> str:
        err = self._lib.sdxGetLastError(self._handles.runtime)
        if not err:
            return ""
        return err.decode("utf-8", errors="replace")

    def close(self) -> None:
        if self._handles.runtime:
            self._lib.sdxDestroyRuntime(self._handles.runtime)
            self._handles.runtime = ctypes.c_void_p()

    def __enter__(self) -> "SdxRuntime":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _raise_on_error(self, status: int, runtime: ctypes.c_void_p, action: str) -> None:
        if status == SDX_STATUS_OK:
            return
        message = ""
        if runtime:
            err = self._lib.sdxGetLastError(runtime)
            if err:
                message = err.decode("utf-8", errors="replace")
        raise RuntimeError(f"{action} failed: status={status}, error={message}")


class SdxModel:
    def __init__(self, runtime: SdxRuntime, model_handle: ctypes.c_void_p) -> None:
        self._runtime = runtime
        self._model_handle = model_handle

    def create_context(self, requested_outputs: Optional[Iterable[str]] = None) -> "SdxContext":
        output_array: Optional[ctypes.Array] = None
        output_ptr = None
        output_count = 0

        if requested_outputs is not None:
            encoded = [name.encode("utf-8") for name in requested_outputs]
            output_count = len(encoded)
            if output_count > 0:
                output_array = (ctypes.c_char_p * output_count)(*encoded)
                output_ptr = ctypes.cast(output_array, ctypes.POINTER(ctypes.c_char_p))

        context = ctypes.c_void_p()
        status = self._runtime._lib.sdxCreateContext(
            self._model_handle,
            output_ptr,
            output_count,
            ctypes.byref(context),
        )
        self._runtime._raise_on_error(status, self._runtime._handles.runtime, "sdxCreateContext")
        return SdxContext(self._runtime, context)

    def close(self) -> None:
        if self._model_handle:
            self._runtime._lib.sdxUnloadModel(self._model_handle)
            self._model_handle = ctypes.c_void_p()

    def __enter__(self) -> "SdxModel":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class SdxContext:
    def __init__(self, runtime: SdxRuntime, context_handle: ctypes.c_void_p) -> None:
        self._runtime = runtime
        self._context_handle = context_handle

    @staticmethod
    def _coerce_tensor_views(tensors: Sequence, is_output: bool):
        views: List[TensorView] = []
        leases: List[_NumpyTensorLease] = []
        for tensor in tensors:
            if isinstance(tensor, TensorView):
                views.append(tensor)
                continue
            if _np is not None and isinstance(tensor, _np.ndarray):
                lease = _NumpyTensorLease(tensor, is_output=is_output)
                leases.append(lease)
                views.append(lease.view)
                continue
            raise TypeError(
                "Expected TensorView or numpy.ndarray entries, got: "
                + type(tensor).__name__
            )

        return (TensorView * len(views))(*views), leases

    def run(self, inputs: Sequence, outputs: Sequence, options: Optional[RunOptions] = None) -> None:
        input_arr, _input_leases = self._coerce_tensor_views(inputs, is_output=False)
        output_arr, output_leases = self._coerce_tensor_views(outputs, is_output=True)

        options_ptr = ctypes.byref(options) if options is not None else None
        status = self._runtime._lib.sdxRun(
            self._context_handle,
            input_arr,
            len(inputs),
            output_arr,
            len(outputs),
            options_ptr,
        )
        self._runtime._raise_on_error(status, self._runtime._handles.runtime, "sdxRun")

        for lease in output_leases:
            lease.finalize()

    def execution_report(self) -> ExecutionReport:
        report = ExecutionReport()
        status = self._runtime._lib.sdxGetExecutionReport(self._context_handle, ctypes.byref(report))
        self._runtime._raise_on_error(status, self._runtime._handles.runtime, "sdxGetExecutionReport")
        return report

    def close(self) -> None:
        if self._context_handle:
            self._runtime._lib.sdxDestroyContext(self._context_handle)
            self._context_handle = ctypes.c_void_p()

    def __enter__(self) -> "SdxContext":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
