#!/usr/bin/env python3
"""Run one platform's existing release matrix outside GitHub Actions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import posixpath
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path


SCCACHE_VERSION = "0.17.0"
SCCACHE_RELEASE_BASE = (
    f"https://github.com/mozilla/sccache/releases/download/v{SCCACHE_VERSION}"
)
ZLUDA_TARGET = "AMD"
ZLUDA_LINUX_LINKER_PACKAGE = "lld"
ZLUDA_LINUX_RPATH_EDITOR_PACKAGE = "patchelf"
ROCM_BUILD_SDKS = {
    "6.2.4": {
        "installer_name": "amdgpu-install_6.2.60204-1_all.deb",
        "installer_url": (
            "https://repo.radeon.com/amdgpu-install/6.2.4/ubuntu/jammy/"
            "amdgpu-install_6.2.60204-1_all.deb"
        ),
        "rocblas_source_url": (
            "https://codeload.github.com/ROCm/rocBLAS/tar.gz/refs/tags/rocm-6.2.4"
        ),
        "tensile_source_url": (
            "https://codeload.github.com/ROCm/Tensile/tar.gz/refs/tags/rocm-6.2.4"
        ),
        "tensile_architectures": "gfx1103",
        # The ROCm binary packages omit Tensile data; these are the generator's
        # pinned Python dependencies, not runtime GPU dependencies.
        "tensile_packages": ("python3-yaml", "python3-msgpack", "python3-joblib"),
        # ROCm 6.2 publishes the ROCt development package separately. Keep
        # it in the managed SDK so the version-matched HSA runtime closure
        # includes libhsakmt.so.1 without relying on the host. Pin the
        # immutable upstream commit instead of the tag archive endpoint: the
        # latter intermittently returned 404 from GitHub Actions even though
        # the tag exists.
        "hsakmt_source_url": (
            "https://codeload.github.com/ROCm/ROCT-Thunk-Interface/tar.gz/"
            "7f307277e71e695dae11e600182a3f5bb53b95b4"
        ),
        # ROCm 6.2 still publishes ROCt/HSAKMT as a standalone repository.
        # Its archive root is the CMake source root and BUILD_SHARED_LIBS
        # already selects the required libhsakmt.so.1 target.
        "hsakmt_source_subdirectory": "",
        "hsakmt_cmake_subdirectory": "",
        "hsakmt_rewrite_static_target": False,
        "hsakmt_disable_static_drm_target": True,
        "component_packages": {
            "hip": (
                "rocm-hip-runtime-dev",
                "hsa-rocr-dev",
                "hsakmt-roct-dev",
                "libnuma-dev",
                "libdrm-dev",
            ),
            # rocblas-dev supplies headers/symlinks; rocblas supplies the
            # runtime library. Tensile data is generated when the packages omit it.
            "rocblas": (
                "rocblas", "rocblas-dev", "python3-yaml", "python3-msgpack",
                "python3-joblib",
            ),
            "hipblaslt": ("hipblaslt-dev",),
            "rocsparse": ("rocsparse-dev",),
            "rocm-smi": ("rocm-smi-lib",),
            "miopen": ("miopen-hip-dev",),
        },
    },
    "7.2.4": {
        "installer_name": "amdgpu-install_7.2.4.70204-1_all.deb",
        "installer_url": (
            "https://repo.radeon.com/amdgpu-install/7.2.4/ubuntu/jammy/"
            "amdgpu-install_7.2.4.70204-1_all.deb"
        ),
        "rocblas_source_url": (
            "https://codeload.github.com/ROCm/rocBLAS/tar.gz/refs/tags/rocm-7.2.4"
        ),
        "tensile_source_url": (
            "https://codeload.github.com/ROCm/Tensile/tar.gz/refs/tags/rocm-7.2.4"
        ),
        "tensile_architectures": "gfx1103",
        "tensile_packages": ("python3-yaml", "python3-msgpack", "python3-joblib"),
        # ROCm 7.2 folded ROCt into the hsa-rocr development package and no
        # longer publishes a standalone hsakmt-roct-dev package.  Keep the
        # matching ROCm source revision here so the managed SDK still contains
        # the user-space libhsakmt.so.1 required by ROCr and ZLUDA.
        "hsakmt_source_url": (
            "https://github.com/ROCm/rocm-systems/archive/refs/tags/"
            "rocm-7.2.4.tar.gz"
        ),
        "hsakmt_source_subdirectory": "projects/rocr-runtime",
        "hsakmt_cmake_subdirectory": "libhsakmt",
        "hsakmt_rewrite_static_target": True,
        "hsakmt_disable_static_drm_target": False,
        "component_packages": {
            # Keep the complete user-space HIP/ROCr/ROCt closure in the
            # versioned SDK archive. The kernel amdgpu/KFD driver remains host-owned.
            "hip": (
                "rocm-hip-runtime-dev",
                "hsa-rocr-dev",
                # ROCt/HSAKMT discovers libnuma and libdrm through CMake at source-build time.
                "libnuma-dev",
                "libdrm-dev",
            ),
            # rocblas-dev supplies headers/symlinks; rocblas supplies the
            # runtime library. Tensile data is generated when the packages omit it.
            "rocblas": (
                "rocblas", "rocblas-dev", "python3-yaml", "python3-msgpack",
                "python3-joblib",
            ),
            "hipblaslt": ("hipblaslt-dev",),
            "rocsparse": ("rocsparse-dev",),
            "rocm-smi": ("rocm-smi-lib",),
            "miopen": ("miopen-hip-dev",),
        },
    },
}
ROCM_BUILD_COMPONENTS = (
    "hip", "rocblas", "hipblaslt", "rocsparse", "rocm-smi", "miopen",
)
DOWNLOAD_RETRIES = 4
TRANSIENT_HTTP_STATUSES = {408, 429, 500, 502, 503, 504}
SDX_MODULES = (
    ":nd4j-sdx-preset",
    ":nd4j-sdx-model",
    ":nd4j-sdx",
    ":nd4j-sdx-litertlm",
)
SDX_ARTIFACT_IDS = tuple(module.removeprefix(":") for module in SDX_MODULES)
SDX_UNCLASSIFIED_ARTIFACT_IDS = SDX_ARTIFACT_IDS
SCCACHE_ASSETS = {
    ("linux", "x86_64"): (
        "x86_64-unknown-linux-musl",
        "67c4a96dd237c1f518f6b36083f270f9976d516f1e57fce891755ea782e50006",
    ),
    ("linux", "arm64"): (
        "aarch64-unknown-linux-musl",
        "821a86343191aa1cbab74bd42f9e93c9a63bf85e4742945f40d3ae84193c1c77",
    ),
    ("macos", "x86_64"): (
        "x86_64-apple-darwin",
        "c2144cafbfe3d22e34ae637f9974ce53613543ac19477fdb287df22ea3668261",
    ),
    ("macos", "arm64"): (
        "aarch64-apple-darwin",
        "0c560bfba31aef5bdfb4fb3d2677f6e61d71c5c00952f2a83344f47aa31f00f1",
    ),
    ("windows", "x86_64"): (
        "x86_64-pc-windows-msvc",
        "caf1932d76a909c909b7a2e41443cdfe3c79a49a380da1a22fa422e1d00d3ca7",
    ),
}


def run(command: list[str], cwd: Path, env: dict[str, str]) -> None:
    print("+", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def bash_command(arguments: list[str], env: dict[str, str]) -> list[str]:
    configured = env.get("DL4J_BASH_EXE")
    if configured:
        return [configured, *arguments]
    if platform.system().lower() == "windows":
        candidates = (
            Path(env.get("ProgramFiles", r"C:\Program Files")) / "Git" / "bin" / "bash.exe",
            Path(r"C:\msys64\usr\bin\bash.exe"),
        )
        for candidate in candidates:
            if candidate.is_file():
                return [str(candidate), *arguments]
        raise RuntimeError(
            "Git Bash is required on Windows; set DL4J_BASH_EXE to bash.exe. "
            "The Windows WSL bash.exe is not supported."
        )
    executable = shutil.which("bash")
    if not executable:
        raise RuntimeError("bash is required to execute release build scripts")
    return [executable, *arguments]


def host_platform() -> tuple[str, str]:
    system = platform.system().lower()
    os_name = {"darwin": "macos", "windows": "windows", "linux": "linux"}.get(system)
    machine = platform.machine().lower()
    architecture = {
        "amd64": "x86_64", "x86_64": "x86_64", "arm64": "arm64", "aarch64": "arm64",
    }.get(machine)
    if os_name is None or architecture is None:
        raise RuntimeError(f"sccache {SCCACHE_VERSION} has no pinned release asset for {system}/{machine}")
    return os_name, architecture


def pinned_sccache_asset() -> tuple[str, str]:
    key = host_platform()
    if key not in SCCACHE_ASSETS:
        raise RuntimeError(
            f"sccache {SCCACHE_VERSION} has no pinned release asset for {key[0]}/{key[1]}"
        )
    target, digest = SCCACHE_ASSETS[key]
    return f"sccache-v{SCCACHE_VERSION}-{target}.tar.gz", digest


def _matching_system_sccache() -> str | None:
    candidate = shutil.which("sccache")
    if not candidate:
        return None
    result = subprocess.run(
        [candidate, "--version"], check=False, capture_output=True, text=True,
    )
    version = (result.stdout + result.stderr).strip()
    return candidate if result.returncode == 0 and f"sccache {SCCACHE_VERSION}" in version else None


def _validate_tar_members(bundle: tarfile.TarFile, destination: Path) -> None:
    root = destination.resolve()
    for member in bundle.getmembers():
        resolved = (destination / member.name).resolve()
        if resolved != root and root not in resolved.parents:
            raise RuntimeError(f"unsafe path in sccache release archive: {member.name!r}")
        if member.issym() or member.islnk() or not (member.isdir() or member.isfile()):
            raise RuntimeError(f"unsafe member in sccache release archive: {member.name!r}")


def ensure_sccache(cache_dir: Path) -> str:
    existing = _matching_system_sccache()
    if existing:
        return existing
    archive_name, expected_digest = pinned_sccache_asset()
    executable_name = "sccache.exe" if host_platform()[0] == "windows" else "sccache"
    executable = cache_dir / "tools" / f"sccache-{SCCACHE_VERSION}" / executable_name
    if executable.is_file():
        return str(executable)
    executable.parent.mkdir(parents=True, exist_ok=True)
    url = f"{SCCACHE_RELEASE_BASE}/{archive_name}"
    print(f"[dl4j-cache] installing pinned sccache {SCCACHE_VERSION} from {url}", flush=True)
    with tempfile.TemporaryDirectory(prefix="sccache-install-", dir=executable.parent.parent) as temporary:
        temporary_root = Path(temporary)
        archive = temporary_root / archive_name
        urllib.request.urlretrieve(url, archive)
        actual_digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        if actual_digest != expected_digest:
            raise RuntimeError(
                f"sccache archive SHA-256 mismatch: expected {expected_digest}, got {actual_digest}"
            )
        extracted = temporary_root / "extracted"
        extracted.mkdir()
        with tarfile.open(archive, mode="r:gz") as bundle:
            _validate_tar_members(bundle, extracted)
            bundle.extractall(extracted)
        candidates = [path for path in extracted.rglob(executable_name) if path.is_file()]
        if len(candidates) != 1:
            raise RuntimeError(
                f"expected one {executable_name} in {archive_name}, found {len(candidates)}"
            )
        staged = executable.with_name(executable.name + ".new")
        shutil.copy2(candidates[0], staged)
        staged.chmod(0o755)
        os.replace(staged, executable)
    return str(executable)


def _activate_managed_llvm_environment(
    env: dict[str, str],
    managed_llvm_root: str,
    managed_llvm_host_tools: str = "",
) -> None:
    """Expose a restored target LLVM package and its native generator tools."""
    env["SD_TRITON_MANAGED_LLVM_ROOT"] = managed_llvm_root
    cmake_args = [
        f"-DSD_TRITON_MANAGED_LLVM_ROOT={managed_llvm_root}",
        "-DSD_TRITON_CONSUMER_KIND=CPU_COMPILER",
    ]
    if managed_llvm_host_tools:
        env["SD_TRITON_MANAGED_LLVM_HOST_TOOLS"] = managed_llvm_host_tools
        cmake_args.append(
            "-DSD_TRITON_MANAGED_LLVM_HOST_TOOLS=" + managed_llvm_host_tools
        )
    existing_cmake_args = env.get("DL4J_CMAKE_ARGS", "").strip()
    env["DL4J_CMAKE_ARGS"] = " ".join(
        item for item in (existing_cmake_args, *cmake_args) if item
    )


def restore_remote_dependency_cache(
    source: Path, config: dict, env: dict[str, str]
) -> None:
    """Restore the managed host/target dependency snapshots before MLIR builds."""
    remote = config.get("compilerCache")
    if not isinstance(remote, dict):
        remote = {}
    snapshots = config.get("dependencyCache")
    if not isinstance(snapshots, dict):
        snapshots = remote.get("dependencyCache")
    if not isinstance(snapshots, dict):
        print("[dl4j-dep-cache] no managed dependency snapshots were advertised", flush=True)
        return
    build = (config.get("shard") or {}).get("build") or {}
    javacpp_platform = str(build.get("javacppPlatform", ""))
    native_backend = str(build.get("backend", ""))
    targets = snapshots.get("targets") or []
    target = next(
        (
            item
            for item in targets
            if isinstance(item, dict)
            and str((item.get("compatibility") or {}).get("javacppPlatform", ""))
            == javacpp_platform
            and str(
                (item.get("compatibility") or {}).get(
                    "nativeBackend",
                    (item.get("compatibility") or {}).get("backend", ""),
                )
            )
            == native_backend
        ),
        None,
    )
    if target is None and len(targets) == 1 and isinstance(targets[0], dict):
        target = targets[0]
    host = snapshots.get("host")
    if not isinstance(host, dict) or not isinstance(target, dict):
        # A managed snapshot is an optimization, not a prerequisite for a
        # release build.  The cache index is populated asynchronously by
        # successful MLIR builds, so a new platform/backend combination can
        # legitimately have no target snapshot yet.  Let the normal release
        # scripts build the dependency locally in that case; failing before
        # the first compiler invocation makes the first build impossible and
        # prevents it from ever seeding the snapshot for subsequent runs.
        print(
            "[dl4j-dep-cache] no exact managed dependency snapshot for "
            f"{javacpp_platform}/{native_backend}; continuing with local dependency build",
            flush=True,
        )
        return

    cache_root = Path.home() / ".libnd4j"
    cache_dir = cache_root / "dep-cache"
    marker = cache_root / ".dl4j-remote-dependency-cache.json"
    identities = {
        "host": str(host.get("identity", "")),
        "target": str(target.get("identity", "")),
    }
    if marker.is_file() and cache_dir.is_dir():
        try:
            cached = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            cached = {}
        managed_root = cached.get("managedLlvmRoot")
        managed_host_tools = cached.get("managedLlvmHostTools", "")
        host_tools_valid = (
            isinstance(managed_host_tools, str)
            and Path(managed_host_tools, "llvm-tblgen").is_file()
            and Path(managed_host_tools, "mlir-tblgen").is_file()
        )
        host_tools_required = javacpp_platform.startswith("android-")
        if (
            cached.get("identities") == identities
            and isinstance(managed_root, str)
            and Path(managed_root).is_dir()
            and (host_tools_valid or not host_tools_required)
        ):
            _activate_managed_llvm_environment(
                env,
                managed_root,
                managed_host_tools if host_tools_valid else "",
            )
            print(
                f"[dl4j-dep-cache] reuse host={identities['host']} "
                f"target={identities['target']} llvm={managed_root} "
                f"hostTools={managed_host_tools if host_tools_valid else 'local-build'}",
                flush=True,
            )
            return

    public_base_url = str(snapshots.get("publicBaseUrl", "")).strip().rstrip("/")
    cloud_io = Path(
        env.get("DL4J_CLOUD_IO", "/opt/dl4j-release/bootstrap/cloud-io.py")
    )
    bucket = ""
    if not public_base_url:
        if not cloud_io.is_file():
            raise RuntimeError(f"managed dependency cache transport is missing: {cloud_io}")
        account = _required_cache_value(remote, "account")
        container = _required_cache_value(remote, "container")
        bucket = f"{account}/{container}"

    def download_object(object_name: str, destination: Path, description: str) -> None:
        if public_base_url:
            download_with_retry(
                append_azure_sas(
                    f"{public_base_url}/{object_name.lstrip('/')}", env
                ),
                destination,
                description,
            )
            return
        run(
            [
                "python3",
                str(cloud_io),
                "download",
                "--bucket",
                bucket,
                "--object",
                object_name,
                "--file",
                str(destination),
                "--client-id",
                env.get("AZURE_CLIENT_ID", ""),
            ],
            source,
            env,
        )
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    restored = []
    package_roots_by_scope = {"host": [], "target": []}
    llvm_host_tool_dirs = []
    with tempfile.TemporaryDirectory(
        prefix="dl4j-dependency-cache-restore-", dir=cache_root
    ) as temporary:
        temporary_root = Path(temporary)
        for scope, manifest in (("host", host), ("target", target)):
            index_object = manifest.get("indexObject")
            archive_object = manifest.get("archiveObject")
            if not isinstance(index_object, str) or not isinstance(archive_object, str):
                raise RuntimeError(f"{scope} dependency snapshot has invalid object metadata")
            index_path = temporary_root / f"{scope}-index.json"
            archive_path = temporary_root / f"{scope}-cache.tar.gz"
            download_object(
                index_object,
                index_path,
                f"{scope} dependency snapshot index",
            )
            index = json.loads(index_path.read_text(encoding="utf-8"))
            indexed_archive = index.get("archiveObject")
            if indexed_archive and indexed_archive != archive_object:
                raise RuntimeError(
                    f"{scope} dependency snapshot index/archive mismatch: "
                    f"{indexed_archive!r} != {archive_object!r}"
                )
            download_object(
                archive_object,
                archive_path,
                f"{scope} dependency snapshot archive",
            )
            expected_size = index.get("size")
            if isinstance(expected_size, int) and archive_path.stat().st_size != expected_size:
                raise RuntimeError(
                    f"{scope} dependency snapshot size mismatch: "
                    f"{archive_path.stat().st_size} != {expected_size}"
                )
            expected_sha256 = index.get("sha256")
            if isinstance(expected_sha256, str) and expected_sha256:
                digest = hashlib.sha256()
                with archive_path.open("rb") as archive_stream:
                    for chunk in iter(lambda: archive_stream.read(8 * 1024 * 1024), b""):
                        digest.update(chunk)
                if digest.hexdigest() != expected_sha256:
                    raise RuntimeError(
                        f"{scope} dependency snapshot SHA-256 mismatch: "
                        f"{digest.hexdigest()} != {expected_sha256}"
                    )
            extracted = temporary_root / f"{scope}-extracted"
            extracted.mkdir()
            with tarfile.open(archive_path, mode="r:gz") as bundle:
                _validate_tar_members(bundle, extracted)
                bundle.extractall(extracted)
            candidate = extracted / "dep-cache"
            if not candidate.is_dir():
                candidates = [
                    path
                    for path in extracted.rglob("dep-cache")
                    if path.is_dir()
                ]
                candidate = candidates[0] if len(candidates) == 1 else extracted
            candidate_roots = []
            for llvm_config in candidate.rglob("LLVMConfig.cmake"):
                root = llvm_config.parents[3]
                if (root / "lib/cmake/mlir/MLIRConfig.cmake").is_file():
                    candidate_roots.append(root.relative_to(candidate))
            candidate_host_tool_dirs = []
            if scope == "host":
                for llvm_tablegen in candidate.rglob("llvm-tblgen"):
                    tool_dir = llvm_tablegen.parent
                    if (tool_dir / "mlir-tblgen").is_file():
                        candidate_host_tool_dirs.append(tool_dir.relative_to(candidate))
            shutil.copytree(candidate, cache_dir, dirs_exist_ok=True)
            package_roots_by_scope[scope].extend(
                cache_dir / relative_root for relative_root in candidate_roots
            )
            llvm_host_tool_dirs.extend(
                cache_dir / relative_root for relative_root in candidate_host_tool_dirs
            )
            restored.append(
                f"{scope}={manifest.get('identity', archive_object)}"
            )
    llvm_roots = sorted(
        {
            root.resolve()
            for roots in package_roots_by_scope.values()
            for root in roots
            if (root / "lib/cmake/llvm/LLVMConfig.cmake").is_file()
            and (root / "lib/cmake/mlir/MLIRConfig.cmake").is_file()
        }
    )
    target_roots = sorted(
        {
            root.resolve()
            for root in package_roots_by_scope["target"]
            if (root / "lib/cmake/llvm/LLVMConfig.cmake").is_file()
            and (root / "lib/cmake/mlir/MLIRConfig.cmake").is_file()
        }
    )
    target_arch = {
        "android-arm64": "AArch64",
        "android-x86_64": "X86",
        "linux-arm64": "AArch64",
        "linux-x86_64": "X86",
    }.get(javacpp_platform)
    def llvm_target_signature(root: Path) -> tuple[str, ...]:
        config_path = root / "lib/cmake/llvm/LLVMConfig.cmake"
        try:
            config_text = config_path.read_text(encoding="utf-8")
        except OSError:
            return ()
        markers = []
        for line in config_text.splitlines():
            stripped = line.strip()
            if stripped.startswith(
                (
                    "set(LLVM_VERSION_",
                    "set(LLVM_PACKAGE_VERSION ",
                    "set(LLVM_TARGETS_TO_BUILD ",
                )
            ):
                markers.append(stripped)
        return tuple(markers)

    platform_target_roots = []
    if target_arch:
        for root in target_roots:
            signature = llvm_target_signature(root)
            if any(
                line.startswith("set(LLVM_TARGETS_TO_BUILD ")
                and target_arch in line
                for line in signature
            ):
                platform_target_roots.append(root)
    selectable_target_roots = platform_target_roots or target_roots
    if len(selectable_target_roots) > 1:
        signatures = {llvm_target_signature(root) for root in selectable_target_roots}
        if len(signatures) == 1 and signatures != {()}:
            # A snapshot can contain duplicate complete packages from repeated
            # builds. They are interchangeable when their LLVM target contract
            # is identical; keep a deterministic root instead of failing.
            selectable_target_roots = [sorted(selectable_target_roots)[0]]
    if len(selectable_target_roots) == 1:
        managed_llvm_root = str(selectable_target_roots[0])
    elif len(llvm_roots) == 1:
        managed_llvm_root = str(llvm_roots[0])
    else:
        raise RuntimeError(
            "managed dependency snapshots did not contain a uniquely selectable "
            f"LLVM/MLIR package (found {len(llvm_roots)} roots: {llvm_roots})"
        )
    selectable_host_tool_dirs = sorted(
        {
            path.resolve()
            for path in llvm_host_tool_dirs
            if (path / "llvm-tblgen").is_file()
            and (path / "mlir-tblgen").is_file()
        }
    )
    managed_llvm_host_tools = (
        str(selectable_host_tool_dirs[0]) if selectable_host_tool_dirs else ""
    )
    _activate_managed_llvm_environment(
        env, managed_llvm_root, managed_llvm_host_tools
    )
    marker.write_text(
        json.dumps(
            {
                "identities": identities,
                "javacppPlatform": javacpp_platform,
                "nativeBackend": native_backend,
                "managedLlvmRoot": managed_llvm_root,
                "managedLlvmHostTools": managed_llvm_host_tools,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(
        "[dl4j-dep-cache] restored " + " ".join(restored) +
        f" llvm={managed_llvm_root} "
        f"hostTools={managed_llvm_host_tools or 'local-build'}",
        flush=True,
    )


def _required_cache_value(settings: dict, name: str) -> str:
    value = settings.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"compilerCache.{name} must be a non-empty string")
    return value.strip()


def _normalize_azure_connection_string(connection_string: str) -> str:
    """Remove endpoint trailing slashes that OpenDAL otherwise duplicates."""
    normalized = []
    for segment in connection_string.split(";"):
        if not segment:
            continue
        name, separator, value = segment.partition("=")
        if separator and name.strip().lower().endswith("endpoint"):
            value = value.rstrip("/")
        normalized.append(f"{name}{separator}{value}")
    return ";".join(normalized)


def _configure_compiler_launchers(env: dict[str, str], compiler_cache: str) -> None:
    env.update({
        "DL4J_COMPILER_CACHE": compiler_cache,
        "SD_REQUIRE_COMPILER_CACHE": "ON",
        "CMAKE_C_COMPILER_LAUNCHER": compiler_cache,
        "CMAKE_CXX_COMPILER_LAUNCHER": compiler_cache,
        "CMAKE_CUDA_COMPILER_LAUNCHER": compiler_cache,
    })


def _activate_sccache(env: dict[str, str], compiler_cache: str) -> None:
    compiler_cache_dir = str(Path(compiler_cache).parent)
    existing_path = env.get("PATH", "")
    env["SD_USE_SCCACHE"] = "1"
    env["PATH"] = (
        compiler_cache_dir
        if not existing_path
        else f"{compiler_cache_dir}{os.pathsep}{existing_path}"
    )
    _configure_compiler_launchers(env, compiler_cache)


def compiler_cache_snapshot_identity(config: dict) -> str | None:
    remote = config.get("compilerCache")
    if not isinstance(remote, dict):
        return None
    snapshot = remote.get("localSnapshot")
    if not isinstance(snapshot, dict):
        return None
    if snapshot.get("schemaVersion") != 1:
        raise ValueError("compilerCache.localSnapshot.schemaVersion must be 1")
    if snapshot.get("name") != "sccache-l0":
        raise ValueError("compilerCache.localSnapshot.name must be 'sccache-l0'")
    shard = config.get("shard")
    if not isinstance(shard, dict):
        raise ValueError("compilerCache.localSnapshot requires a shard contract")
    shard_id = str(shard.get("id", "")).strip()
    contract_digest = str(shard.get("contractDigest", "")).strip()
    if not shard_id or not contract_digest:
        raise ValueError(
            "compilerCache.localSnapshot requires shard.id and shard.contractDigest"
        )
    operating_system, architecture = host_platform()
    return toolchain_cache_identity(
        "sccache-l0",
        {
            "snapshotSchemaVersion": 1,
            "sccacheVersion": SCCACHE_VERSION,
            "platform": operating_system,
            "architecture": architecture,
            "backend": _required_cache_value(remote, "backend"),
            "keyPrefix": _required_cache_value(remote, "keyPrefix"),
            "shard": shard_id,
            "shardContractDigest": contract_digest,
        },
    )


def restore_compiler_cache_snapshot(
    config: dict, env: dict[str, str], cache_dir: Path
) -> None:
    identity = compiler_cache_snapshot_identity(config)
    if identity is None or toolchain_cache_transport(config, env) is None:
        return
    started = time.monotonic()
    restored = restore_toolchain_dependency(
        config,
        env,
        name="sccache-l0",
        identity=identity,
        destination=cache_dir,
    )
    duration = round(time.monotonic() - started, 3)
    env.update({
        "DL4J_SCCACHE_SNAPSHOT_IDENTITY": identity,
        "DL4J_SCCACHE_SNAPSHOT_DIR": str(cache_dir),
        "DL4J_SCCACHE_SNAPSHOT_RESTORED": "true" if restored else "false",
        "DL4J_SCCACHE_SNAPSHOT_RESTORE_SECONDS": str(duration),
    })
    print(
        f"[dl4j-cache-prefetch] name=sccache-l0 identity={identity} "
        f"status={'hit' if restored else 'miss'} durationSeconds={duration}",
        flush=True,
    )


def compiler_cache_snapshot_metrics(env: dict[str, str]) -> dict:
    identity = env.get("DL4J_SCCACHE_SNAPSHOT_IDENTITY")
    if not identity:
        return {"enabled": False}
    return {
        "enabled": True,
        "identity": identity,
        "restoreStatus": (
            "hit" if env.get("DL4J_SCCACHE_SNAPSHOT_RESTORED") == "true" else "miss"
        ),
        "restoreDurationSeconds": float(
            env.get("DL4J_SCCACHE_SNAPSHOT_RESTORE_SECONDS", "0")
        ),
    }


def directory_size_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def publish_compiler_cache_snapshot(config: dict, env: dict[str, str]) -> dict:
    metrics = compiler_cache_snapshot_metrics(env)
    if not metrics["enabled"]:
        return metrics
    if metrics["restoreStatus"] == "hit":
        metrics["publishStatus"] = "not-required"
        return metrics
    cache_dir = Path(env["DL4J_SCCACHE_SNAPSHOT_DIR"])
    expanded_bytes = directory_size_bytes(cache_dir)
    started = time.monotonic()
    publish_toolchain_dependency(
        config,
        env,
        name="sccache-l0",
        identity=metrics["identity"],
        source=cache_dir,
    )
    duration = round(time.monotonic() - started, 3)
    metrics.update({
        "publishStatus": "published",
        "publishDurationSeconds": duration,
        "expandedBytes": expanded_bytes,
    })
    print(
        f"[dl4j-cache-prefetch] name=sccache-l0 identity={metrics['identity']} "
        f"status=published expandedBytes={expanded_bytes} durationSeconds={duration}",
        flush=True,
    )
    return metrics


def configure_compiler_cache(
    config: dict, source: Path, env: dict[str, str]
) -> tuple[str | None, bool]:
    remote = config.get("compilerCache")
    if remote is not None and not isinstance(remote, dict):
        raise ValueError("compilerCache must be an object")
    if remote is not None:
        backend = _required_cache_value(remote, "backend")
        if backend not in {"s3", "gcs", "azure"}:
            raise ValueError(f"unsupported compilerCache.backend {backend!r}")
        configured_cache_dir = env.get("DL4J_SCCACHE_DIR", "").strip()
        cache_dir = (
            Path(configured_cache_dir)
            if configured_cache_dir
            else source.parent / "sccache"
        )
        if configured_cache_dir and not cache_dir.is_absolute():
            raise ValueError("DL4J_SCCACHE_DIR must be an absolute path")
        restore_compiler_cache_snapshot(config, env, cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        compiler_cache = ensure_cached_sccache(cache_dir, config, env)
        env.update({
            "SCCACHE_DIR": str(cache_dir),
            "SCCACHE_CACHE_SIZE": "100G",
            "SCCACHE_IDLE_TIMEOUT": "0",
            "SCCACHE_BASEDIRS": str(source.resolve()),
            "SCCACHE_ERROR_LOG": str(cache_dir / "sccache-error.log"),
            "SCCACHE_MULTILEVEL_CHAIN": f"disk,{backend}",
            # A remote cache outage must not block compiler wrappers or make the
            # hosted runner appear dead; the local cache remains authoritative.
            "SCCACHE_MULTILEVEL_WRITE_ERROR_POLICY": "l0",
        })
        prefix = _required_cache_value(remote, "keyPrefix")
        if backend == "s3":
            env.update({
                "SCCACHE_BUCKET": _required_cache_value(remote, "bucket"),
                "SCCACHE_REGION": _required_cache_value(remote, "region"),
                "SCCACHE_S3_KEY_PREFIX": prefix,
                "SCCACHE_S3_SERVER_SIDE_ENCRYPTION": "true",
            })
        elif backend == "gcs":
            env.update({
                "SCCACHE_GCS_BUCKET": _required_cache_value(remote, "bucket"),
                "SCCACHE_GCS_KEY_PREFIX": prefix,
                "SCCACHE_GCS_RW_MODE": "READ_WRITE",
            })
        else:
            connection_string = str(remote.get("connectionString", "")).strip()
            connection_string_env = str(remote.get("connectionStringEnv", "")).strip()
            if connection_string_env:
                connection_string = env.get(connection_string_env, "").strip()
                if not connection_string:
                    raise ValueError(
                        f"compilerCache environment variable {connection_string_env!r} is empty"
                    )
            if not connection_string:
                raise ValueError(
                    "compilerCache.connectionString or compilerCache.connectionStringEnv is required"
                )
            env.update({
                "SCCACHE_AZURE_CONNECTION_STRING": _normalize_azure_connection_string(
                    connection_string
                ),
                "SCCACHE_AZURE_BLOB_CONTAINER": _required_cache_value(remote, "container"),
                "SCCACHE_AZURE_KEY_PREFIX": prefix,
            })
        print(
            f"[dl4j-cache] backend={backend} mode=disk+remote keyPrefix={prefix}", flush=True,
        )
        _activate_sccache(env, compiler_cache)
        run([compiler_cache, "--start-server"], source, env)
        return compiler_cache, True

    compiler_cache = shutil.which("sccache") or shutil.which("ccache")
    if not compiler_cache:
        return None, False
    cache_name = "sccache" if Path(compiler_cache).stem.lower() == "sccache" else "ccache"
    cache_dir = source.parent / cache_name
    cache_dir.mkdir(parents=True, exist_ok=True)
    if cache_name == "ccache":
        _configure_compiler_launchers(env, compiler_cache)
        env.update({
            "CCACHE_DIR": str(cache_dir), "CCACHE_BASEDIR": str(source),
            "CCACHE_NOHASHDIR": "true", "CCACHE_MAXSIZE": "100G",
        })
        run([compiler_cache, "--zero-stats"], source, env)
        return compiler_cache, False
    _activate_sccache(env, compiler_cache)
    env.update({
        "SCCACHE_DIR": str(cache_dir), "SCCACHE_CACHE_SIZE": "100G",
        "SCCACHE_IDLE_TIMEOUT": "0",
    })
    run([compiler_cache, "--start-server"], source, env)
    return compiler_cache, True


def variant_platform_extension(variant: dict) -> str:
    """Return the exact JavaCPP platform extension declared by a release variant."""
    return str(variant.get("platformExtension", variant.get("suffix", "")))


def variant_libnd4j_classifier(build: dict, variant: dict) -> str:
    """Return the exact libnd4j classifier declared by a release variant."""
    classifier_suffix = variant.get("classifierSuffix", variant.get("suffix", ""))
    return f"{build['javacppPlatform']}{classifier_suffix}"


def variant_flags(build: dict, variant: dict) -> list[str]:
    backend = build["backend"]
    helper = variant.get("helper", "")
    extension = variant.get("extension", "")
    platform_extension = variant_platform_extension(variant)
    flags = [f"-Dlibnd4j.classifier={variant_libnd4j_classifier(build, variant)}"]
    if platform_extension:
        flags.append(f"-Djavacpp.platform.extension={platform_extension}")
    if helper:
        flags.append(f"-Dlibnd4j.helper={helper}")
    if extension:
        flags.append(f"-Dlibnd4j.extension={extension}")
    if variant.get("mlir"):
        if variant.get("triton", True):
            flags.append("-Dlibnd4j.triton=ON")
        helpers = variant.get("helpers", ["mlir"])
        flags.extend([f"-Dlibnd4j.helpers={','.join(helpers)}", "-Dlibnd4j.mlir=ON"])
    elif variant.get("triton"):
        flags.append("-Dlibnd4j.triton=ON")
    if helper == "mps":
        flags.extend(["-Dlibnd4j.triton=ON", "-Dlibnd4j.helper=mps"])
    if backend == "cuda":
        flags.extend(["-Dlibnd4j.chip=cuda", "-Dlibnd4j.cuda.compile.skip=false", "-Dlibnd4j.cpu.compile.skip=true"])
    return flags


def variant_artifact_classifier(build: dict, variant: dict) -> str:
    """Return the attached JavaCPP JAR classifier for a release variant."""
    return f"{build['javacppPlatform']}{variant_platform_extension(variant)}"


def sdx_variant_artifact_classifier(build: dict, variant: dict) -> str:
    """Return the SDX classifier without colliding with another backend's JAR."""
    classifier = variant_artifact_classifier(build, variant)
    backend = build.get("backend")
    if backend == "cpu" or build.get("zludaVersion"):
        return classifier
    if backend == "cuda" and build.get("cudaVersion"):
        return (
            f"{build['javacppPlatform']}-cuda-{build['cudaVersion']}"
            f"{variant_platform_extension(variant)}"
        )
    if backend == "vulkan":
        return f"{build['javacppPlatform']}-vulkan{variant_platform_extension(variant)}"
    return classifier


def has_base_platform_variant(build: dict) -> bool:
    """Return whether this selected slice owns the unextended platform output."""
    return any(
        variant_artifact_classifier(build, variant) == build["javacppPlatform"]
        for variant in build.get("variants", [])
    )


def sdx_enabled_for_build(build: dict) -> bool:
    """Return whether this desktop backend owns the opt-in SDX component."""
    return (
        "sdx" in build.get("profiles", [])
        and build.get("backend") in {"cpu", "cuda", "vulkan"}
        and build.get("javacppPlatform") in {
            "linux-x86_64", "windows-x86_64", "macosx-arm64"
        }
    )


def sdx_native_configuration(
    source: Path, build: dict, variant: dict
) -> tuple[str, str, Path]:
    """Return the backend library, linker target, and central native output path."""
    backend = build.get("backend")
    if backend == "cuda":
        return (
            "nd4jcuda",
            "nd4jcuda",
            source / "libnd4j/blasbuild/cuda",
        )
    if backend == "vulkan":
        return (
            "nd4jvulkan",
            "nd4jvulkan",
            source
            / "libnd4j/blasbuild/vulkan"
            / variant_libnd4j_classifier(build, variant),
        )
    return (
        "nd4jcpu",
        "nd4jcpu",
        source / "libnd4j/blasbuild/cpu",
    )


def variant_artifact_classifier_for(
    build: dict, variant: dict, artifact_id: str
) -> str:
    """Use the SDX classifier only for SDX artifacts; backend JARs keep their contract."""
    if artifact_id in SDX_ARTIFACT_IDS:
        return sdx_variant_artifact_classifier(build, variant)
    return variant_artifact_classifier(build, variant)


def enable_sdx_release_component(build: dict, rules: dict) -> None:
    """Make the declarative SDX profile publishable without changing other lanes."""
    if not sdx_enabled_for_build(build):
        return
    build["modules"] = list(dict.fromkeys([*build.get("modules", []), *SDX_MODULES]))
    if rules.get("mode", "all") != "classifier":
        return
    rules["artifactIds"] = list(dict.fromkeys([
        *rules.get("artifactIds", []), *SDX_ARTIFACT_IDS
    ]))
    rules["unclassifiedArtifactIds"] = list(dict.fromkeys([
        *rules.get("unclassifiedArtifactIds", []), *SDX_UNCLASSIFIED_ARTIFACT_IDS
    ]))
    sdx_classifier_tokens = [
        sdx_variant_artifact_classifier(build, variant)
        for variant in build.get("variants", [])
    ]
    rules["classifierTokens"] = list(dict.fromkeys([
        *rules.get("classifierTokens", []), *sdx_classifier_tokens
    ]))


def required_classifier_artifact_ids(build: dict, rules: dict) -> tuple[str, ...]:
    """Return the published runtime artifacts that must carry each classifier."""
    if rules.get("mode", "all") != "classifier":
        return ()
    backend = build.get("backend")
    if backend == "cpu":
        required = ("nd4j-native", "nd4j-native-preset")
    elif backend == "cuda":
        cuda_version = str(build.get("cudaVersion", ""))
        if not cuda_version:
            raise ValueError("CUDA classifier validation requires cudaVersion")
        if build.get("zludaVersion"):
            required = (
                f"nd4j-zluda-{cuda_version}",
                f"nd4j-cuda-{cuda_version}-preset",
            )
        else:
            primary = f"nd4j-cuda-{cuda_version}"
            required = (primary, f"{primary}-preset")
    elif backend == "vulkan":
        # nd4j-vulkan owns the native classifier. nd4j-vulkan-preset contains
        # platform-neutral JavaCPP declarations and remains unclassified.
        required = ("nd4j-vulkan",)
    else:
        return ()
    if sdx_enabled_for_build(build):
        # nd4j-sdx is the only SDX artifact with a platform classifier; the
        # model, preset, and optional provider remain unclassified Java jars.
        required = (*required, "nd4j-sdx")
    owned = set(rules.get("artifactIds", []))
    missing_owned = [artifact_id for artifact_id in required if artifact_id not in owned]
    if missing_owned:
        raise ValueError(
            "classifier artifact rules do not own required artifacts: "
            + ", ".join(missing_owned)
        )
    modules = {
        str(module).removeprefix(":") for module in build.get("modules", [])
    }
    missing_modules = [artifact_id for artifact_id in required if artifact_id not in modules]
    if missing_modules:
        raise ValueError(
            "classifier build does not include required modules: "
            + ", ".join(missing_modules)
        )
    return required


def exact_classifier_jar_candidates(
    repository: Path, artifact_id: str, version: str, classifier: str
) -> tuple[Path, ...]:
    file_name = f"{artifact_id}-{version}-{classifier}.jar"
    return tuple(
        repository / namespace / artifact_id / version / file_name
        for namespace in (
            Path("org/eclipse/deeplearning4j"),
            Path("org/nd4j"),
        )
    )


def required_unclassified_artifact_ids(build: dict, rules: dict) -> tuple[str, ...]:
    """Return explicitly owned runtime artifacts that must be unclassified JARs."""
    required = tuple(dict.fromkeys(rules.get("unclassifiedArtifactIds", [])))
    if not required:
        return ()
    if rules.get("mode", "all") != "classifier":
        raise ValueError("unclassifiedArtifactIds require classifier artifact mode")
    owned = set(rules.get("artifactIds", []))
    missing_owned = [artifact_id for artifact_id in required if artifact_id not in owned]
    if missing_owned:
        raise ValueError(
            "unclassifiedArtifactIds must be a subset of artifactIds: "
            + ", ".join(missing_owned)
        )
    modules = {
        str(module).removeprefix(":") for module in build.get("modules", [])
    }
    missing_modules = [artifact_id for artifact_id in required if artifact_id not in modules]
    if missing_modules:
        raise ValueError(
            "unclassified artifact build does not include required modules: "
            + ", ".join(missing_modules)
        )
    return required


def exact_unclassified_jar_candidates(
    repository: Path, artifact_id: str, version: str
) -> tuple[Path, ...]:
    file_name = f"{artifact_id}-{version}.jar"
    return tuple(
        repository / namespace / artifact_id / version / file_name
        for namespace in (
            Path("org/eclipse/deeplearning4j"),
            Path("org/nd4j"),
        )
    )


def attest_classifier_archive_contract(
    path: Path,
    rules: dict,
    artifact_id: str,
    classifier: str,
    phase: str,
) -> None:
    """Fail closed when a classified JAR omits its declared native closure."""
    contracts = rules.get("classifierArchiveContracts", {}) or {}
    contract = contracts.get(artifact_id)
    if contract is None:
        return
    if not isinstance(contract, dict):
        raise ValueError(
            f"classifierArchiveContracts[{artifact_id!r}] must be an object"
        )

    def expand(value: object) -> str:
        return str(value).format(classifier=classifier)

    required_entries = tuple(
        expand(entry) for entry in contract.get("requiredEntries", []) or []
    )
    manifest_entry_value = contract.get("runtimeManifest")
    manifest_entry = (
        expand(manifest_entry_value) if manifest_entry_value is not None else None
    )
    required_runtime_aliases_value = contract.get("requiredRuntimeAliases", {}) or {}
    if not isinstance(required_runtime_aliases_value, dict):
        raise ValueError(
            f"classifierArchiveContracts[{artifact_id!r}].requiredRuntimeAliases "
            "must be an object"
        )
    required_runtime_aliases = {
        expand(alias): expand(target)
        for alias, target in required_runtime_aliases_value.items()
    }
    if not required_entries or not manifest_entry:
        raise ValueError(
            f"classifierArchiveContracts[{artifact_id!r}] must declare "
            "requiredEntries and runtimeManifest"
        )

    try:
        with zipfile.ZipFile(path) as archive:
            entries = {item.filename: item for item in archive.infolist()}
            missing = [entry for entry in required_entries if entry not in entries]
            empty = [
                entry for entry in required_entries
                if entry in entries and entries[entry].file_size == 0
            ]
            if missing or empty:
                details = []
                if missing:
                    details.append("missing " + ", ".join(missing))
                if empty:
                    details.append("empty " + ", ".join(empty))
                raise RuntimeError(
                    f"{phase} classifier archive {path.name} violates its "
                    f"runtime contract: {'; '.join(details)}"
                )
            try:
                manifest_text = archive.read(manifest_entry).decode("utf-8")
            except UnicodeDecodeError as exc:
                raise RuntimeError(
                    f"{phase} classifier archive {path.name} has a non-UTF-8 "
                    f"runtime manifest {manifest_entry}"
                ) from exc
            manifest_lines = [
                line.strip() for line in manifest_text.splitlines() if line.strip()
            ]
            runtime_names = [
                line for line in manifest_lines if not line.startswith("#")
            ]
            if not runtime_names:
                raise RuntimeError(
                    f"{phase} classifier archive {path.name} has an empty "
                    f"runtime closure manifest {manifest_entry}"
                )
            alias_count = None
            runtime_aliases = {}
            for line in manifest_lines:
                if line.startswith("# runtime-alias-count="):
                    if alias_count is not None:
                        raise RuntimeError(
                            f"{phase} classifier archive {path.name} has duplicate "
                            "runtime alias counts"
                        )
                    try:
                        alias_count = int(line.partition("=")[2])
                    except ValueError as exc:
                        raise RuntimeError(
                            f"{phase} classifier archive {path.name} has an invalid "
                            f"runtime alias count: {line}"
                        ) from exc
                elif line.startswith("# runtime-alias="):
                    mapping = line.partition("=")[2]
                    alias, separator, target = mapping.partition("->")
                    alias = alias.strip()
                    target = target.strip()
                    if not separator or not alias or not target:
                        raise RuntimeError(
                            f"{phase} classifier archive {path.name} has an invalid "
                            f"runtime alias declaration: {line}"
                        )
                    if alias in runtime_aliases:
                        raise RuntimeError(
                            f"{phase} classifier archive {path.name} has duplicate "
                            f"runtime alias declarations for {alias}"
                        )
                    runtime_aliases[alias] = target
            if alias_count is not None and alias_count != len(runtime_aliases):
                raise RuntimeError(
                    f"{phase} classifier archive {path.name} declares "
                    f"{alias_count} runtime aliases but lists {len(runtime_aliases)}"
                )
            missing_required_aliases = [
                f"{alias}->{target}"
                for alias, target in required_runtime_aliases.items()
                if runtime_aliases.get(alias) != target
            ]
            if missing_required_aliases:
                raise RuntimeError(
                    f"{phase} classifier archive {path.name} does not declare "
                    "required runtime aliases: " + ", ".join(missing_required_aliases)
                )
            unsafe = [
                name
                for name in (
                    runtime_names
                    + list(runtime_aliases)
                    + list(runtime_aliases.values())
                )
                if "/" in name or "\\" in name or name in {".", ".."}
            ]
            if unsafe:
                raise RuntimeError(
                    f"{phase} classifier archive {path.name} has unsafe runtime "
                    f"manifest entries: {', '.join(unsafe)}"
                )
            invalid_alias_targets = [
                f"{alias}->{target}"
                for alias, target in runtime_aliases.items()
                if target not in runtime_names or alias in runtime_names
            ]
            if invalid_alias_targets:
                raise RuntimeError(
                    f"{phase} classifier archive {path.name} has runtime aliases "
                    "which do not resolve directly to canonical manifest runtimes: "
                    + ", ".join(invalid_alias_targets)
                )
            manifest_parent = manifest_entry.rpartition("/")[0]
            def manifest_sibling(name):
                return f"{manifest_parent}/{name}" if manifest_parent else name

            runtime_entries = [manifest_sibling(name) for name in runtime_names]
            missing_runtime = [
                entry for entry in runtime_entries
                if entry not in entries or entries[entry].file_size == 0
            ]
            if missing_runtime:
                raise RuntimeError(
                    f"{phase} classifier archive {path.name} is missing "
                    "manifest-owned runtimes: " + ", ".join(missing_runtime)
                )
            alias_entries = {
                manifest_sibling(alias): manifest_sibling(target)
                for alias, target in runtime_aliases.items()
            }
            missing_aliases = [
                alias
                for alias in alias_entries
                if alias not in entries or entries[alias].file_size == 0
            ]
            if missing_aliases:
                raise RuntimeError(
                    f"{phase} classifier archive {path.name} is missing "
                    "manifest-declared runtime aliases: " + ", ".join(missing_aliases)
                )
            mismatched_aliases = [
                f"{alias}->{target}"
                for alias, target in alias_entries.items()
                if (
                    entries[alias].file_size != entries[target].file_size
                    or entries[alias].CRC != entries[target].CRC
                )
            ]
            if mismatched_aliases:
                raise RuntimeError(
                    f"{phase} classifier archive {path.name} has runtime aliases "
                    "whose content differs from the canonical runtime: "
                    + ", ".join(mismatched_aliases)
                )
    except zipfile.BadZipFile as exc:
        raise RuntimeError(
            f"{phase} classifier artifact is not a readable JAR: {path}"
        ) from exc

    print(
        f"[dl4j-attestation] phase={phase} classifier={classifier} "
        f"artifact={artifact_id} archive-entries={len(entries)} "
        f"runtime-closure={len(runtime_names)} "
        f"runtime-aliases={len(runtime_aliases)}",
        flush=True,
    )


def reset_unclassified_artifacts(
    repository: Path, build: dict, rules: dict, version: str | None
) -> None:
    required = required_unclassified_artifact_ids(build, rules)
    if not required:
        return
    if not version:
        raise ValueError("unclassified artifact validation requires a release version")
    for artifact_id in required:
        for path in exact_unclassified_jar_candidates(repository, artifact_id, version):
            path.unlink(missing_ok=True)


def attest_unclassified_artifacts(
    repository: Path,
    build: dict,
    rules: dict,
    version: str | None,
    phase: str,
) -> None:
    required = required_unclassified_artifact_ids(build, rules)
    if not required:
        return
    if not version:
        raise ValueError("unclassified artifact validation requires a release version")
    found = {
        artifact_id: [
            path
            for path in exact_unclassified_jar_candidates(
                repository, artifact_id, version
            )
            if path.is_file()
        ]
        for artifact_id in required
    }
    missing = [artifact_id for artifact_id, paths in found.items() if not paths]
    if missing:
        raise RuntimeError(
            f"{phase} is missing exact unclassified JARs: {', '.join(missing)}"
        )
    paths = sorted(
        path.relative_to(repository).as_posix()
        for matches in found.values()
        for path in matches
    )
    print(
        f"[dl4j-attestation] phase={phase} "
        f"unclassified-artifacts={','.join(paths)}",
        flush=True,
    )


def reset_variant_classifier_artifacts(
    repository: Path, build: dict, rules: dict, variant: dict, version: str | None
) -> None:
    required = required_classifier_artifact_ids(build, rules)
    if not required:
        return
    if not version:
        raise ValueError("classifier artifact validation requires a release version")
    for artifact_id in required:
        classifier = variant_artifact_classifier_for(build, variant, artifact_id)
        for path in exact_classifier_jar_candidates(
            repository, artifact_id, version, classifier
        ):
            path.unlink(missing_ok=True)


def attest_variant_classifier_artifacts(
    repository: Path,
    build: dict,
    rules: dict,
    variant: dict,
    version: str | None,
    phase: str,
) -> None:
    required = required_classifier_artifact_ids(build, rules)
    if not required:
        return
    if not version:
        raise ValueError("classifier artifact validation requires a release version")
    found: dict[str, list[Path]] = {}
    for artifact_id in required:
        classifier = variant_artifact_classifier_for(build, variant, artifact_id)
        found[artifact_id] = [
            path
            for path in exact_classifier_jar_candidates(
                repository, artifact_id, version, classifier
            )
            if path.is_file()
        ]
    missing = [artifact_id for artifact_id, paths in found.items() if not paths]
    if missing:
        missing_classifiers = ", ".join(
            f"{artifact_id}={variant_artifact_classifier_for(build, variant, artifact_id)}"
            for artifact_id in missing
        )
        raise RuntimeError(
            f"{phase} is missing exact classifier JARs for variant "
            f"{variant['name']}: {missing_classifiers}"
        )
    for artifact_id, paths_for_artifact in found.items():
        classifier = variant_artifact_classifier_for(build, variant, artifact_id)
        for path in paths_for_artifact:
            attest_classifier_archive_contract(
                path, rules, artifact_id, classifier, phase
            )
    paths = sorted(
        path.relative_to(repository).as_posix()
        for matches in found.values()
        for path in matches
    )
    print(
        f"[dl4j-attestation] phase={phase} variant={variant['name']} "
        f"classifier={variant_artifact_classifier(build, variant)} "
        f"artifacts={','.join(paths)}",
        flush=True,
    )


def build_cross_platform(source: Path, build: dict, repository: Path, env: dict[str, str]) -> None:
    platform = build["javacppPlatform"]
    cross_env = env.copy()
    if platform in {"linux-arm64", "macosx-arm64"} and Path("/opt/protoc-21.7/bin").exists():
        cross_env["PATH"] = f"/opt/protoc-21.7/bin:{cross_env.get('PATH', '')}"
    cross_env.update({
        "DL4J_PLATFORM": platform,
        "DL4J_OS": "windows" if platform == "windows-x86_64" else ("macos" if platform == "macosx-arm64" else "linux"),
        "DL4J_MAVEN_GOAL": "install",
        "DL4J_MAVEN_REPOSITORY": str(repository),
    })
    script = source / "build-scripts/release/cross-platform.sh"
    run(bash_command([str(script), "--run-tokenizers"], cross_env), source, cross_env)
    run(bash_command([str(script), "--run-java"], cross_env), source, cross_env)


def prepare_openblas(
    source: Path,
    build: dict,
    env: dict[str, str],
    config: dict | None = None,
) -> None:
    if ":libnd4j" not in build.get("modules", []):
        return
    classifier = build["javacppPlatform"]
    version = "0.3.28-1.5.11"
    archive = source / f"openblas-{version}-{classifier}.jar"
    url = f"https://repo1.maven.org/maven2/org/bytedeco/openblas/{version}/{archive.name}"
    target = source / "openblas_home"
    identity = toolchain_cache_identity(
        "openblas",
        {"version": version, "classifier": classifier, "url": url},
    )
    restored = restore_toolchain_dependency(
        config or {},
        env,
        name="openblas",
        identity=identity,
        destination=target,
    )
    if not restored:
        print(f"+ download {url}", flush=True)
        download_with_retry(url, archive, f"OpenBLAS {version} {classifier}")
        with zipfile.ZipFile(archive) as bundle:
            root = target.resolve()
            for member in bundle.infolist():
                destination = (target / member.filename).resolve()
                if destination != root and root not in destination.parents:
                    raise RuntimeError(
                        f"unsafe path in OpenBLAS archive: {member.filename!r}"
                    )
            bundle.extractall(target)
    headers = list(target.rglob("include/cblas.h"))
    if not headers:
        raise RuntimeError(f"OpenBLAS archive has no include/cblas.h: {archive}")
    env["OPENBLAS_PATH"] = str(headers[0].parent.parent)
    if not restored:
        publish_toolchain_dependency(
            config or {},
            env,
            name="openblas",
            identity=identity,
            source=target,
        )


def prepare_prebuilt_libnd4j(source: Path, build: dict, env: dict[str, str]) -> str:
    """Restore an optional libnd4j archive before JavaCPP packaging."""
    url = str(build.get("libnd4jUrl", "")).strip()
    if not url:
        return ""
    tools = source / ".dl4j-release-tools"
    tools.mkdir(parents=True, exist_ok=True)
    archive = tools / "prebuilt-libnd4j.zip"
    print(f"[dl4j-phase] phase=prebuilt-libnd4j status=started url={url}", flush=True)
    download_with_retry(url, archive, "prebuilt libnd4j")
    with tempfile.TemporaryDirectory(prefix="dl4j-prebuilt-libnd4j-") as temporary:
        extracted = Path(temporary)
        with zipfile.ZipFile(archive) as bundle:
            root = extracted.resolve()
            for member in bundle.infolist():
                destination = (extracted / member.filename).resolve()
                if destination != root and root not in destination.parents:
                    raise RuntimeError(
                        f"unsafe path in prebuilt libnd4j archive: {member.filename!r}"
                    )
            bundle.extractall(extracted)
        candidates = [
            path for path in extracted.rglob("blasbuild")
            if path.is_dir() and path.parent.name == "libnd4j"
        ]
        if len(candidates) != 1:
            raise RuntimeError(
                "prebuilt libnd4j archive must contain exactly one libnd4j/blasbuild "
                f"directory; found {len(candidates)}"
            )
        target = source / "libnd4j/blasbuild"
        target.mkdir(parents=True, exist_ok=True)
        shutil.copytree(candidates[0], target, dirs_exist_ok=True)

    backend = "cuda" if build.get("backend") == "cuda" else "cpu"
    engine = "ENGINE_CUDA" if backend == "cuda" else "ENGINE_CPU"
    config_text = (
        "#ifndef LIBND4J_CONFIG_H\n"
        "#define LIBND4J_CONFIG_H\n"
        f"#define DEFAULT_ENGINE samediff::{engine}\n"
        "#endif\n"
    )
    for config_path in (
        source / "libnd4j/include/config.h",
        source / f"libnd4j/blasbuild/{backend}/include/config.h",
    ):
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(config_text, encoding="utf-8")
    generated = source / "libnd4j/include/generated/include_ops.h"
    generated.parent.mkdir(parents=True, exist_ok=True)
    generated.touch(exist_ok=True)

    if backend == "cpu":
        blas = source / "libnd4j/blasbuild/cpu/blas"
        openblas = Path(env["OPENBLAS_PATH"]) / "lib"
        if blas.is_dir():
            openblas.mkdir(parents=True, exist_ok=True)
            for path in blas.iterdir():
                destination = openblas / path.name
                if path.is_dir():
                    shutil.copytree(path, destination, dirs_exist_ok=True)
                else:
                    shutil.copy2(path, destination)
    print("[dl4j-phase] phase=prebuilt-libnd4j status=complete", flush=True)
    return url


def urlopen_with_retry(request: urllib.request.Request, description: str):
    delay = 1.0
    for attempt in range(DOWNLOAD_RETRIES):
        try:
            return urllib.request.urlopen(request, timeout=60)
        except urllib.error.HTTPError as exc:
            if exc.code not in TRANSIENT_HTTP_STATUSES or attempt + 1 == DOWNLOAD_RETRIES:
                raise RuntimeError(f"{description} failed with HTTP {exc.code}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt + 1 == DOWNLOAD_RETRIES:
                raise RuntimeError(f"{description} failed: {exc}") from exc
        print(f"[dl4j-download] {description} attempt {attempt + 1} failed; retrying in {delay:g}s", flush=True)
        time.sleep(delay)
        delay = min(delay * 2, 8.0)
    raise AssertionError("unreachable")


def download_with_retry(url: str, destination: Path, description: str) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "dl4j-release-builder"})
    try:
        with urlopen_with_retry(request, description) as response, destination.open("wb") as output:
            shutil.copyfileobj(response, output)
    except BaseException:
        destination.unlink(missing_ok=True)
        raise


def append_azure_sas(url: str, env: dict[str, str]) -> str:
    """Authenticate public dependency snapshots from GitHub with the shared SAS."""
    connection = (
        env.get("DL4J_AZURE_CONNECTION_STRING")
        or env.get("SCCACHE_AZURE_CONNECTION_STRING")
        or ""
    )
    sas = next(
        (
            part.split("=", 1)[1].lstrip("?")
            for part in connection.split(";")
            if part.startswith("SharedAccessSignature=")
        ),
        "",
    )
    if not sas:
        return url
    return url + ("&" if "?" in url else "?") + sas


def toolchain_cache_identity(name: str, contract: dict) -> str:
    payload = json.dumps(
        {"schemaVersion": 1, "name": name, "contract": contract},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def toolchain_cache_transport(
    config: dict, env: dict[str, str]
) -> tuple[Path, str, str, str | None] | None:
    remote = config.get("compilerCache")
    if not isinstance(remote, dict):
        return None
    cache = remote.get("toolchainCache")
    if not isinstance(cache, dict) or cache.get("schemaVersion") != 1:
        return None
    helper = Path(env.get("DL4J_DEPENDENCY_CACHE_HELPER", ""))
    cloud_io = Path(env.get("DL4J_CLOUD_IO", ""))
    account = str(remote.get("account", ""))
    container = str(remote.get("container", ""))
    prefix = str(cache.get("keyPrefix", "")).strip("/")
    if (
        not helper.is_file()
        or not cloud_io.is_file()
        or not account
        or not container
        or not prefix
    ):
        return None
    client_id = str(config.get("managedIdentityClientId", "")) or None
    return helper, f"{account}/{container}", prefix, client_id


def restore_toolchain_dependency(
    config: dict,
    env: dict[str, str],
    *,
    name: str,
    identity: str,
    destination: Path,
) -> bool:
    transport = toolchain_cache_transport(config, env)
    if transport is None:
        return False
    helper, bucket, prefix, client_id = transport
    command = [
        sys.executable,
        str(helper),
        "restore",
        "--cloud-io",
        env["DL4J_CLOUD_IO"],
        "--bucket",
        bucket,
        "--prefix",
        prefix,
        "--name",
        name,
        "--identity",
        identity,
        "--destination",
        str(destination),
    ]
    if client_id:
        command.extend(["--client-id", client_id])
    result = subprocess.run(command, env=env, check=False)
    if result.returncode == 0:
        return True
    if result.returncode == 3:
        return False
    raise RuntimeError(
        f"managed toolchain cache restore failed for {name}/{identity}: "
        f"exit code {result.returncode}"
    )


def publish_toolchain_dependency(
    config: dict,
    env: dict[str, str],
    *,
    name: str,
    identity: str,
    source: Path,
) -> None:
    transport = toolchain_cache_transport(config, env)
    if transport is None:
        return
    helper, bucket, prefix, client_id = transport
    command = [
        sys.executable,
        str(helper),
        "publish",
        "--cloud-io",
        env["DL4J_CLOUD_IO"],
        "--bucket",
        bucket,
        "--prefix",
        prefix,
        "--name",
        name,
        "--identity",
        identity,
        "--source",
        str(source),
    ]
    if client_id:
        command.extend(["--client-id", client_id])
    subprocess.run(command, env=env, check=True)


def ensure_cached_sccache(
    cache_dir: Path, config: dict, env: dict[str, str]
) -> str:
    existing = _matching_system_sccache()
    if existing:
        return existing
    archive_name, archive_digest = pinned_sccache_asset()
    operating_system, architecture = host_platform()
    executable_name = "sccache.exe" if operating_system == "windows" else "sccache"
    destination = cache_dir / "tools" / f"sccache-{SCCACHE_VERSION}"
    identity = toolchain_cache_identity(
        "sccache",
        {
            "version": SCCACHE_VERSION,
            "platform": operating_system,
            "architecture": architecture,
            "archive": archive_name,
            "archiveSha256": archive_digest,
        },
    )
    restored = (destination / executable_name).is_file()
    if not restored:
        restored = restore_toolchain_dependency(
            config,
            env,
            name="sccache",
            identity=identity,
            destination=destination,
        )
    executable = ensure_sccache(cache_dir)
    if not restored:
        publish_toolchain_dependency(
            config,
            env,
            name="sccache",
            identity=identity,
            source=destination,
        )
    return executable


def _prepend_environment_path(env: dict[str, str], name: str, value: str) -> None:
    entries = [entry for entry in env.get(name, "").split(os.pathsep) if entry]
    env[name] = os.pathsep.join([value] + [entry for entry in entries if entry != value])


def rocm_build_spec(build: dict) -> dict | None:
    """Return the pinned CPU-hosted ROCm SDK contract for a Linux ZLUDA build."""
    if not build.get("zludaVersion") or not build.get("javacppPlatform", "").startswith("linux-"):
        return None
    version = build.get("rocmVersion")
    if version not in ROCM_BUILD_SDKS:
        raise ValueError(
            f"Linux ZLUDA releases require a supported rocmVersion; got {version!r}"
        )
    if build.get("rocmBuildOnly") is not True:
        raise ValueError("Linux ZLUDA releases must declare rocmBuildOnly=true")
    components = tuple(build.get("rocmBuildComponents", ()))
    if components != ROCM_BUILD_COMPONENTS:
        raise ValueError(
            "Linux ZLUDA releases require the exact ROCm build components "
            f"{list(ROCM_BUILD_COMPONENTS)!r}; got {list(components)!r}"
        )
    sdk = ROCM_BUILD_SDKS[version]
    packages = tuple(
        package
        for component in components
        for package in sdk["component_packages"][component]
    )
    return {
        "version": version,
        "components": components,
        "packages": packages,
        "installer_name": sdk["installer_name"],
        "installer_url": sdk["installer_url"],
        "rocblas_source_url": sdk["rocblas_source_url"],
        "tensile_source_url": sdk["tensile_source_url"],
        "tensile_architectures": sdk["tensile_architectures"],
        "tensile_packages": sdk["tensile_packages"],
        "hsakmt_source_url": sdk["hsakmt_source_url"],
        "hsakmt_source_subdirectory": sdk["hsakmt_source_subdirectory"],
        "hsakmt_cmake_subdirectory": sdk["hsakmt_cmake_subdirectory"],
        "hsakmt_rewrite_static_target": sdk["hsakmt_rewrite_static_target"],
        "hsakmt_disable_static_drm_target": sdk.get("hsakmt_disable_static_drm_target", False),
    }


def _first_existing_file(root: Path, relative_paths: tuple[str, ...]) -> Path | None:
    for relative_path in relative_paths:
        candidate = root / relative_path
        if candidate.is_file():
            return candidate
    return None


def attest_rocm_build_toolchain(
        build: dict,
        env: dict[str, str],
        root: Path | None = None,
        emit: bool = True) -> dict[str, Path] | None:
    """Fail closed on the SDK files needed to build, without probing GPU hardware."""
    spec = rocm_build_spec(build)
    if spec is None:
        return None
    rocm_root = root or Path(env.get("ROCM_PATH", "/opt/rocm"))
    version_file = rocm_root / ".info/version"
    hip_header = rocm_root / "include/hip/hip_runtime.h"
    hipcc = rocm_root / "bin/hipcc"
    hip_runtime = _first_existing_file(rocm_root, (
        "lib/libamdhip64.so",
        "lib64/libamdhip64.so",
        "lib/x86_64-linux-gnu/libamdhip64.so",
    ))
    hsa_runtime = _first_existing_file(rocm_root, (
        "lib/libhsa-runtime64.so.1",
        "lib64/libhsa-runtime64.so.1",
        "lib/x86_64-linux-gnu/libhsa-runtime64.so.1",
    ))
    hsakmt_runtime = _first_existing_file(rocm_root, (
        "lib/libhsakmt.so.1",
        "lib64/libhsakmt.so.1",
        "lib/x86_64-linux-gnu/libhsakmt.so.1",
    ))
    miopen_header = rocm_root / "include/miopen/miopen.h"
    miopen_runtime = _first_existing_file(rocm_root, (
        "lib/libMIOpen.so",
        "lib64/libMIOpen.so",
        "lib/x86_64-linux-gnu/libMIOpen.so",
    ))
    rocblas_header = rocm_root / "include/rocblas/rocblas.h"
    rocblas_runtime = _first_existing_file(rocm_root, (
        "lib/librocblas.so",
        "lib64/librocblas.so",
        "lib/x86_64-linux-gnu/librocblas.so",
    ))
    hipblaslt_runtime = _first_existing_file(rocm_root, (
        "lib/libhipblaslt.so",
        "lib64/libhipblaslt.so",
        "lib/x86_64-linux-gnu/libhipblaslt.so",
    ))
    rocsparse_runtime = _first_existing_file(rocm_root, (
        "lib/librocsparse.so",
        "lib64/librocsparse.so",
        "lib/x86_64-linux-gnu/librocsparse.so",
    ))
    rocm_smi_runtime = _first_existing_file(rocm_root, (
        "lib/librocm_smi64.so",
        "lib64/librocm_smi64.so",
        "lib/x86_64-linux-gnu/librocm_smi64.so",
    ))
    failures = []
    installed_version = (
        version_file.read_text(encoding="utf-8").strip()
        if version_file.is_file() else ""
    )
    if not installed_version.startswith(str(spec["version"])):
        failures.append(
            f"{version_file} does not attest ROCm {spec['version']} "
            f"(found {installed_version or 'missing'})"
        )
    for description, path in (
            ("HIP header", hip_header),
            ("HIP compiler driver", hipcc)):
        if not path.is_file():
            failures.append(f"{description} is missing at {path}")
    if hip_runtime is None:
        failures.append(f"HIP runtime library is missing below {rocm_root}")
    if hsa_runtime is None:
        failures.append(f"HSA runtime library is missing below {rocm_root}")
    if hsakmt_runtime is None:
        failures.append(f"HSAKMT runtime library is missing below {rocm_root}")
    if "rocblas" in spec["components"]:
        if not rocblas_header.is_file():
            failures.append(f"rocBLAS header is missing at {rocblas_header}")
        if rocblas_runtime is None:
            failures.append(f"rocBLAS runtime library is missing below {rocm_root}")
    for component, description, runtime in (
            ("hipblaslt", "hipBLASLt", hipblaslt_runtime),
            ("rocsparse", "rocSPARSE", rocsparse_runtime),
            ("rocm-smi", "ROCm SMI", rocm_smi_runtime)):
        if component in spec["components"] and runtime is None:
            failures.append(f"{description} runtime library is missing below {rocm_root}")
    if "miopen" in spec["components"]:
        if not miopen_header.is_file():
            failures.append(f"MIOpen header is missing at {miopen_header}")
        if miopen_runtime is None:
            failures.append(f"MIOpen runtime library is missing below {rocm_root}")
    if failures:
        raise RuntimeError("ROCm build-toolchain attestation failed: " + "; ".join(failures))

    env["ROCM_PATH"] = str(rocm_root)
    env["ROCM_HOME"] = str(rocm_root)
    env["HIP_PATH"] = str(rocm_root)
    env["DL4J_ZLUDA_REQUIRE_ROCM"] = "1"
    env["DL4J_ZLUDA_REQUIRE_MIOPEN"] = "1"
    _prepend_environment_path(env, "PATH", str(rocm_root / "bin"))
    _prepend_environment_path(env, "LD_LIBRARY_PATH", str(hip_runtime.parent))
    _prepend_environment_path(env, "CPLUS_INCLUDE_PATH", str(rocm_root / "include"))
    attested = {
        "version": version_file,
        "hipHeader": hip_header,
        "hipcc": hipcc,
        "hipRuntime": hip_runtime,
        "hsaRuntime": hsa_runtime,
        "hsakmtRuntime": hsakmt_runtime,
        "rocblasHeader": rocblas_header,
        "rocblasRuntime": rocblas_runtime,
        "hipblasltRuntime": hipblaslt_runtime,
        "rocsparseRuntime": rocsparse_runtime,
        "rocmSmiRuntime": rocm_smi_runtime,
        "miopenHeader": miopen_header,
        "miopenRuntime": miopen_runtime,
    }
    if emit:
        print(
            "[dl4j-attestation] "
            f"rocmVersion={spec['version']} rocmBuildOnly=true "
            f"components={','.join(spec['components'])} root={rocm_root} "
            f"hipHeader={hip_header} hipRuntime={hip_runtime} "
            f"hsaRuntime={hsa_runtime} hsakmtRuntime={hsakmt_runtime} "
            f"rocblasHeader={rocblas_header} rocblasRuntime={rocblas_runtime} "
            f"hipblasltRuntime={hipblaslt_runtime} "
            f"rocsparseRuntime={rocsparse_runtime} rocmSmiRuntime={rocm_smi_runtime} "
            f"miopenHeader={miopen_header} miopenRuntime={miopen_runtime} "
            "hardwareProbe=skipped",
            flush=True,
        )
    return attested


def rocm_tensile_data_file(root: Path) -> Path | None:
    """Locate a generated rocBLAS Tensile master file below a ROCm prefix."""
    if not root.is_dir():
        return None
    candidates = sorted(
        path for path in root.rglob("TensileLibrary.dat") if path.is_file()
    )
    return candidates[0] if candidates else None


def extract_rocm_source_archive(archive_path: Path, destination: Path) -> None:
    """Safely extract a pinned ROCm source archive."""
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as archive:
        root = destination.resolve()
        for member in archive.getmembers():
            member_path = Path(member.name)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise RuntimeError(
                    f"unsafe ROCm source archive member: {member.name!r}"
                )
            member_destination = (destination / member.name).resolve()
            if member_destination != root and root not in member_destination.parents:
                raise RuntimeError(
                    f"unsafe ROCm source archive member: {member.name!r}"
                )
            if member.issym() or member.islnk():
                link_target = posixpath.normpath(
                    posixpath.join(posixpath.dirname(member.name), member.linkname)
                )
                if link_target == ".." or link_target.startswith("../"):
                    raise RuntimeError(
                        f"unsafe ROCm source archive link: {member.name!r} -> "
                        f"{member.linkname!r}"
                    )
        archive.extractall(destination)


def build_rocm_tensile_data(
    build: dict,
    spec: dict,
    rocm_root: Path,
    env: dict[str, str],
    temporary_directory: Path,
) -> Path:
    """Generate the missing rocBLAS Tensile data for the target AMD ISA."""
    existing = rocm_tensile_data_file(rocm_root)
    if existing is not None:
        return existing
    if platform.system().lower() != "linux" or platform.machine().lower() not in {
        "amd64", "x86_64"
    }:
        raise RuntimeError("Tensile data generation requires a Linux x86_64 builder")

    rocblas_archive = temporary_directory / "rocblas-source.tar.gz"
    tensile_archive = temporary_directory / "tensile-source.tar.gz"
    download_with_retry(
        str(spec["rocblas_source_url"]), rocblas_archive,
        f"ROCm {spec['version']} rocBLAS source",
    )
    download_with_retry(
        str(spec["tensile_source_url"]), tensile_archive,
        f"ROCm {spec['version']} Tensile source",
    )
    rocblas_source = temporary_directory / "rocblas-source"
    tensile_source = temporary_directory / "tensile-source"
    extract_rocm_source_archive(rocblas_archive, rocblas_source)
    extract_rocm_source_archive(tensile_archive, tensile_source)

    logic_candidates = sorted(
        path for path in rocblas_source.rglob("Logic")
        if path.is_dir() and path.parent.name == "Tensile"
        and path.parent.parent.name == "blas3"
    )
    tool_candidates = sorted(
        path for path in tensile_source.rglob("TensileCreateLibrary")
        if path.is_file() and path.parent.name == "bin"
    )
    if len(logic_candidates) != 1 or len(tool_candidates) != 1:
        raise RuntimeError(
            "ROCm source archives must contain exactly one rocBLAS Tensile Logic "
            f"directory and generator; found {len(logic_candidates)} and "
            f"{len(tool_candidates)}"
        )

    generator_environment = env.copy()
    generator_environment["ROCM_PATH"] = str(rocm_root)
    generator_environment["ROCM_HOME"] = str(rocm_root)
    generator_environment["HIP_PATH"] = str(rocm_root)
    _prepend_environment_path(generator_environment, "PATH", str(rocm_root / "bin"))
    _prepend_environment_path(
        generator_environment, "LD_LIBRARY_PATH", str(rocm_root / "lib")
    )
    output = temporary_directory / "tensile-output"
    threads = str(max(1, int(build.get("buildThreads") or 4)))
    run(
        [
            sys.executable,
            str(tool_candidates[0]),
            f"--architecture={spec['tensile_architectures']}",
            "--no-enumerate",
            f"--jobs={threads}",
            "--cxx-compiler=hipcc",
            "--library-format=msgpack",
            str(logic_candidates[0]),
            str(output),
            "HIP",
        ],
        temporary_directory,
        generator_environment,
    )

    generated = sorted(
        path for path in output.rglob("TensileLibrary.dat") if path.is_file()
    )
    if len(generated) != 1:
        raise RuntimeError(
            "TensileCreateLibrary did not produce exactly one TensileLibrary.dat; "
            f"found {len(generated)}"
        )
    generated_root = generated[0].parent
    destination = rocm_root / "lib" / "rocblas" / "library"
    destination.mkdir(parents=True, exist_ok=True)
    for source_path in generated_root.rglob("*"):
        if not source_path.is_file():
            continue
        target_path = destination / source_path.relative_to(generated_root)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)
    installed = destination / "TensileLibrary.dat"
    if not installed.is_file():
        raise RuntimeError(
            f"Tensile data generation completed without {installed}"
        )
    print(
        f"[dl4j-rocm] generated Tensile data for "
        f"{spec['tensile_architectures']}: {installed}",
        flush=True,
    )
    return installed


def rocm_hsakmt_source_candidates(extracted: Path, spec: dict) -> list[Path]:
    """Locate the one declared ROCt CMake source root in an extracted archive."""
    source_subdirectory = str(spec["hsakmt_source_subdirectory"]).strip("/")
    source_pattern = (
        f"*/{source_subdirectory}/CMakeLists.txt"
        if source_subdirectory
        else "*/CMakeLists.txt"
    )
    return sorted(extracted.glob(source_pattern))


def disable_rocm_hsakmt_static_target(cmake_text: str) -> str:
    """Guard an upstream static-only DRM target during shared runtime builds."""
    start_marker = "## Create separate target file for static builds\n"
    end_marker = "\n###########################\n# Packaging directives\n"
    start = cmake_text.find(start_marker)
    if start < 0:
        raise RuntimeError(
            "version-matched ROCt source does not expose the expected static "
            "HSAKMT target block"
        )
    end = cmake_text.find(end_marker, start + len(start_marker))
    if end < 0:
        raise RuntimeError(
            "version-matched ROCt source does not expose the static HSAKMT "
            "target block terminator"
        )
    if cmake_text[:start].endswith("if ( NOT BUILD_SHARED_LIBS)\n"):
        return cmake_text
    block = cmake_text[start:end]
    if "${HSAKMT_STATIC_DRM_TARGET}" not in block:
        raise RuntimeError(
            "version-matched ROCt static target block is missing its expected "
            "DRM target"
        )
    if "if ( NOT BUILD_SHARED_LIBS)" in block:
        return cmake_text
    return (
        cmake_text[:start]
        + "if ( NOT BUILD_SHARED_LIBS)\n"
        + block
        + "endif()\n"
        + cmake_text[end:]
    )


def build_rocm_hsakmt(
    build: dict,
    spec: dict,
    rocm_root: Path,
    env: dict[str, str],
    temporary_directory: Path,
) -> Path:
    """Build the version-matched ROCt thunk when the ROCm SDK omits it."""
    existing = _first_existing_file(rocm_root, (
        "lib/libhsakmt.so.1",
        "lib64/libhsakmt.so.1",
        "lib/x86_64-linux-gnu/libhsakmt.so.1",
    ))
    if existing is not None:
        return existing
    if platform.system().lower() != "linux" or platform.machine().lower() not in {
        "amd64", "x86_64"
    }:
        raise RuntimeError("managed ROCt bootstrap requires a Linux x86_64 builder")
    source_archive = temporary_directory / "roct-source.tar.gz"
    download_with_retry(
        str(spec["hsakmt_source_url"]),
        source_archive,
        f"ROCm {spec['version']} ROCt source",
    )
    extracted = temporary_directory / "roct-source"
    extracted.mkdir(parents=True, exist_ok=True)
    with tarfile.open(source_archive, "r:gz") as archive:
        root = extracted.resolve()
        for member in archive.getmembers():
            member_path = Path(member.name)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise RuntimeError(f"unsafe ROCm source archive member: {member.name!r}")
            destination = (extracted / member.name).resolve()
            if destination != root and root not in destination.parents:
                raise RuntimeError(f"unsafe ROCm source archive member: {member.name!r}")
            if member.issym() or member.islnk():
                # Upstream ROCt source archives may contain harmless in-tree
                # links. Validate their normalized target before extraction so
                # the archive can retain those links without permitting a link
                # to escape the extraction root.
                link_target = posixpath.normpath(
                    posixpath.join(posixpath.dirname(member.name), member.linkname)
                )
                if link_target == ".." or link_target.startswith("../"):
                    raise RuntimeError(
                        f"unsafe ROCm source archive link: {member.name!r} -> "
                        f"{member.linkname!r}"
                    )
        archive.extractall(extracted)
    source_candidates = rocm_hsakmt_source_candidates(extracted, spec)
    if len(source_candidates) != 1:
        raise RuntimeError(
            "ROCm source archive must contain exactly one ROCt source tree; "
            f"found {len(source_candidates)}"
        )
    source = source_candidates[0].parent
    # Some ROCm releases publish the ROCt source with a static-only target
    # even when BUILD_SHARED_LIBS=ON. ZLUDA needs the version-matched user-space thunk
    # as a shared object so the HSA runtime can resolve libhsakmt.so.1 at load
    # time. Adapt only the downloaded build recipe; the repository source stays
    # untouched and the resulting library remains beneath ROCM_PATH.
    hsakmt_cmake_subdirectory = str(spec["hsakmt_cmake_subdirectory"]).strip("/")
    hsakmt_cmake = (
        source / hsakmt_cmake_subdirectory / "CMakeLists.txt"
        if hsakmt_cmake_subdirectory
        else source / "CMakeLists.txt"
    )
    if not hsakmt_cmake.is_file():
        raise RuntimeError(
            f"version-matched ROCt source is missing {hsakmt_cmake.relative_to(source)}"
        )
    hsakmt_cmake_text = None
    if (
        spec["hsakmt_rewrite_static_target"]
        or spec["hsakmt_disable_static_drm_target"]
    ):
        hsakmt_cmake_text = hsakmt_cmake.read_text(encoding="utf-8")
        if spec["hsakmt_disable_static_drm_target"]:
            adapted = disable_rocm_hsakmt_static_target(hsakmt_cmake_text)
            if adapted != hsakmt_cmake_text:
                hsakmt_cmake_text = adapted
                print(
                    "[dl4j-rocm] disabling upstream static HSAKMT DRM target "
                    "for shared runtime build",
                    flush=True,
                )
        if spec["hsakmt_rewrite_static_target"]:
            static_target = 'add_library (${HSAKMT_TARGET} STATIC "")'
            shared_target = 'add_library (${HSAKMT_TARGET} SHARED "")'
            if static_target not in hsakmt_cmake_text:
                raise RuntimeError(
                    "version-matched ROCt source no longer exposes the expected "
                    "static HSAKMT target; refusing to create an unverified runtime"
                )
            hsakmt_cmake_text = hsakmt_cmake_text.replace(static_target, shared_target, 1)
            print(
                "[dl4j-rocm] adapting downloaded ROCt target to shared libhsakmt.so.1",
                flush=True,
            )
        hsakmt_cmake.write_text(hsakmt_cmake_text, encoding="utf-8")
    cmake_build = temporary_directory / "rocr-runtime-build"
    threads = str(max(1, int(build.get("buildThreads") or env.get("DL4J_BUILD_THREADS", "4"))))
    cmake_environment = env.copy()
    cmake_environment["ROCM_PATH"] = str(rocm_root)
    cmake_environment["ROCM_HOME"] = str(rocm_root)
    try:
        rocm_major, rocm_minor, rocm_patch = (
            int(part) for part in str(spec["version"]).split(".")
        )
    except (TypeError, ValueError):
        raise RuntimeError(
            f"ROCm version must be semantic major.minor.patch, got {spec['version']!r}"
        ) from None
    rocm_patch_version = f"{rocm_major}{rocm_minor:02d}{rocm_patch:02d}"
    run(
        [
            "cmake", "-S", str(source), "-B", str(cmake_build),
            "-DBUILD_ROCR=OFF",
            "-DBUILD_SHARED_LIBS=ON",
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_INSTALL_PREFIX={rocm_root}",
            "-DCMAKE_INSTALL_LIBDIR=lib",
            "-DENABLE_LDCONFIG=OFF",
            "-DHSAKMT_WERROR=OFF",
            "-DADDRESS_SANITIZER=OFF",
            f"-DROCM_PATCH_VERSION={rocm_patch_version}",
        ],
        source,
        cmake_environment,
    )
    run(
        ["cmake", "--build", str(cmake_build), "--target", "hsakmt", "--parallel", threads],
        source,
        cmake_environment,
    )
    run(["cmake", "--install", str(cmake_build)], source, cmake_environment)
    installed = _first_existing_file(rocm_root, (
        "lib/libhsakmt.so.1",
        "lib64/libhsakmt.so.1",
        "lib/x86_64-linux-gnu/libhsakmt.so.1",
    ))
    if installed is None:
        raise RuntimeError(
            f"ROCt bootstrap completed without libhsakmt.so.1 below {rocm_root}"
        )
    print(f"[dl4j-rocm] built managed HSAKMT runtime: {installed}", flush=True)
    return installed


def prepare_rocm_build_toolchain(
    build: dict, env: dict[str, str], config: dict | None = None
) -> None:
    """Restore or install a pinned ROCm SDK without requiring AMD hardware."""
    config = config or {}
    spec = rocm_build_spec(build)
    if spec is None:
        return

    os_contract = ""
    os_release = Path("/etc/os-release")
    if os_release.is_file():
        os_contract = os_release.read_text(encoding="utf-8")
    # ROCM_PATH is the explicit version selector. When it is not supplied,
    # use the versioned prefix created by the ROCm repository installer rather
    # than assuming that the installer also creates an /opt/rocm symlink.
    rocm_root = Path(env.get("ROCM_PATH", f"/opt/rocm-{spec['version']}"))
    cache_identity = toolchain_cache_identity(
        "rocm-sdk",
        {
            "platform": platform.system().lower(),
            "architecture": platform.machine().lower(),
            "osRelease": os_contract,
            "version": spec["version"],
            "components": list(spec["components"]),
            "packages": list(spec["packages"]),
            "runtimeClosure": [
                "libamdhip64.so",
                "libhsa-runtime64.so.1",
                "libhsakmt.so.1",
            ],
            "installerUrl": spec["installer_url"],
            "hsakmtSourceUrl": spec["hsakmt_source_url"],
            "rocblasSourceUrl": spec["rocblas_source_url"],
            "tensileSourceUrl": spec["tensile_source_url"],
            "tensileArchitectures": spec["tensile_architectures"],
            "hsakmtSourceSubdirectory": spec["hsakmt_source_subdirectory"],
            "hsakmtCmakeSubdirectory": spec["hsakmt_cmake_subdirectory"],
            "hsakmtRewriteStaticTarget": spec["hsakmt_rewrite_static_target"],
            "destination": str(rocm_root),
        },
    )
    rocm_ready = True
    cache_seed_required = False
    try:
        attest_rocm_build_toolchain(build, env, root=rocm_root, emit=False)
        if rocm_tensile_data_file(rocm_root) is None:
            raise RuntimeError("rocBLAS Tensile data is missing")
    except RuntimeError:
        rocm_ready = restore_toolchain_dependency(
            config,
            env,
            name="rocm-sdk",
            identity=cache_identity,
            destination=rocm_root,
        )
        cache_seed_required = not rocm_ready
        if rocm_ready:
            try:
                attest_rocm_build_toolchain(
                    build, env, root=rocm_root, emit=False
                )
                if rocm_tensile_data_file(rocm_root) is None:
                    raise RuntimeError("rocBLAS Tensile data is missing")
            except RuntimeError:
                # Older cache entries may predate the managed ROCt closure.
                # Treat them as a seed candidate and repair them in place.
                rocm_ready = False
                cache_seed_required = True

    linker = shutil.which("ld.lld", path=env.get("PATH"))
    rpath_editor = shutil.which("patchelf", path=env.get("PATH"))
    if not rocm_ready or linker is None or rpath_editor is None:
        if platform.system().lower() != "linux" or platform.machine().lower() not in {
                "amd64", "x86_64"}:
            raise RuntimeError(
                "ROCm build-only provisioning requires a Linux x86_64 builder"
            )
        if not hasattr(os, "geteuid") or os.geteuid() != 0:
            raise RuntimeError(
                "ROCm build-only provisioning requires root inside the disposable build container"
            )

        install_env = env.copy()
        install_env["DEBIAN_FRONTEND"] = "noninteractive"
        with tempfile.TemporaryDirectory(prefix="dl4j-rocm-sdk-") as temporary_directory:
            installer = Path(temporary_directory) / str(spec["installer_name"])
            if not rocm_ready:
                download_with_retry(
                    str(spec["installer_url"]),
                    installer,
                    f"ROCm {spec['version']} Ubuntu Jammy repository installer",
                )
            run(["apt-get", "update"], Path("/"), install_env)
            if not rocm_ready:
                run([
                    "apt-get", "install", "-y", "--no-install-recommends", str(installer),
                ], Path("/"), install_env)
                run(["apt-get", "update"], Path("/"), install_env)

            packages = list(spec["packages"]) if not rocm_ready else []
            if linker is None:
                packages.append(ZLUDA_LINUX_LINKER_PACKAGE)
            if rpath_editor is None:
                packages.append(ZLUDA_LINUX_RPATH_EDITOR_PACKAGE)
            if packages:
                run([
                    "apt-get", "install", "-y", "--no-install-recommends", *packages,
                ], Path("/"), install_env)
            build_rocm_hsakmt(
                build,
                spec,
                rocm_root,
                install_env,
                Path(temporary_directory),
            )

    if rocm_tensile_data_file(rocm_root) is None:
        with tempfile.TemporaryDirectory(prefix="dl4j-tensile-") as temporary_directory:
            build_rocm_tensile_data(
                build, spec, rocm_root, env, Path(temporary_directory)
            )

    attest_rocm_build_toolchain(build, env, root=rocm_root)
    if cache_seed_required:
        publish_toolchain_dependency(
            config,
            env,
            name="rocm-sdk",
            identity=cache_identity,
            source=rocm_root.resolve(),
        )
    linker = shutil.which("ld.lld", path=env.get("PATH"))
    if linker is None:
        raise RuntimeError(
            "ZLUDA build-only provisioning requires ld.lld from the lld package"
        )
    env["DL4J_ZLUDA_LINKER"] = linker
    rpath_editor = shutil.which("patchelf", path=env.get("PATH"))
    if rpath_editor is None:
        raise RuntimeError(
            "ZLUDA build-only provisioning requires patchelf for relocatable "
            "classifier RUNPATHs"
        )
    env["DL4J_ZLUDA_PATCHELF"] = rpath_editor
    print(
        f"[dl4j-attestation] zludaLinker={linker} linkerFamily=lld "
        f"rpathEditor={rpath_editor} runtimeRunpath=$ORIGIN "
        "sectionGc=true hardwareProbe=skipped",
        flush=True,
    )


def zluda_platform(build: dict) -> str:
    platform = build.get("javacppPlatform", "")
    if platform.startswith("windows-"):
        return "windows"
    if platform.startswith("linux-"):
        return "linux"
    raise ValueError(f"ZLUDA releases do not support JavaCPP platform {platform!r}")


def package_runtime_sdk(source: Path, output: Path, threads: int) -> int:
    produced = 0
    for cache in source.glob("libnd4j/blasbuild/*/CMakeCache.txt"):
        run(["cmake", "--build", str(cache.parent), "--target", "sdx_runtime_bindings", "--parallel", str(threads)], source, os.environ.copy())
    for asset in source.glob("libnd4j/blasbuild/*/sdx-runtime-sdk/dist/**/*"):
        if not asset.is_file() or asset.suffix.lower() not in {".zip", ".aar"}:
            continue
        relative = asset.relative_to(source)
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(asset, destination)
        produced += 1
    for bundle in source.glob("libnd4j/blasbuild/*/sdx-runtime-sdk/dist/**/*.xcframework"):
        if not bundle.is_dir():
            continue
        relative = bundle.relative_to(source)
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.make_archive(str(destination), "zip", root_dir=bundle.parent, base_dir=bundle.name)
        produced += 1
    return produced


def package_sdk_jars(repository: Path, output: Path, build: dict, rules: dict) -> int:
    artifact_ids = set(rules.get("artifactIds", []))
    if not artifact_ids:
        return 0
    classifiers = {
        variant_artifact_classifier(build, variant)
        for variant in build["variants"]
    }
    if sdx_enabled_for_build(build):
        classifiers.update(
            sdx_variant_artifact_classifier(build, variant)
            for variant in build["variants"]
        )
    produced = 0
    for namespace in (Path("org/eclipse/deeplearning4j"), Path("org/nd4j")):
        root = repository / namespace
        if not root.exists():
            continue
        for jar in root.rglob("*.jar"):
            relative = jar.relative_to(root)
            artifact_id = relative.parts[0] if relative.parts else ""
            if artifact_id not in artifact_ids or jar.name.endswith(("-sources.jar", "-javadoc.jar", "-tests.jar")):
                continue
            classified = any(classifier in jar.name for classifier in classifiers)
            platform_jar = any(token in jar.name for token in ("linux-", "windows-", "macosx-", "android-", "ios-"))
            if platform_jar and not classified:
                continue
            destination = output / "jars" / jar.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(jar, destination)
            produced += 1
    return produced


def build_aot(source: Path, output: Path, build: dict, repository: Path, env: dict[str, str]) -> int:
    if not build.get("buildAot"):
        return 0
    if not has_base_platform_variant(build):
        print(
            "[dl4j-phase] phase=aot status=skipped "
            "reason=no-base-platform-variant",
            flush=True,
        )
        return 0
    graalvm_home = env.get("GRAALVM_HOME")
    native_image = Path(graalvm_home) / "bin/native-image" if graalvm_home else None
    if not native_image or not native_image.exists():
        raise RuntimeError("buildAot is enabled but GRAALVM_HOME/bin/native-image is unavailable")
    aot_env = env.copy()
    aot_env["JAVA_HOME"] = graalvm_home
    aot_env["PATH"] = str(native_image.parent) + os.pathsep + aot_env.get("PATH", "")
    platform = build["javacppPlatform"]
    variant = "cuda" if build["backend"] == "cuda" else "cpu"
    profiles = "sdx-aot,native,cuda" if variant == "cuda" else "sdx-aot,native"
    run([
        "mvn", "--batch-mode", f"-P{profiles}", "-pl", ":sdx-aot", "-DskipTests",
        f"-Dmaven.repo.local={repository}", f"-Djavacpp.platform={platform}",
        "-Djavacpp.platform.extension=", "package",
    ], source, aot_env)
    produced = 0
    for asset in source.glob("nd4j/sdx-aot/target/sdx-aot-*-aot.zip"):
        destination = output / asset.name
        shutil.copy2(asset, destination)
        produced += 1
    return produced


def stage_repository(repository: Path, output: Path, rules: dict) -> None:
    namespaces = (Path("org/eclipse/deeplearning4j"), Path("org/nd4j"))
    mode = rules.get("mode", "all")
    artifact_ids = set(rules.get("artifactIds", []))
    classifier_tokens = tuple(rules.get("classifierTokens", []))
    unclassified_artifact_ids = set(rules.get("unclassifiedArtifactIds", []))
    if not unclassified_artifact_ids.issubset(artifact_ids):
        raise ValueError("unclassifiedArtifactIds must be a subset of artifactIds")
    include_metadata = bool(rules.get("includeMetadata", False))
    for namespace in namespaces:
        root = repository / namespace
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or "maven-metadata" in path.name:
                continue
            relative_under_namespace = path.relative_to(root)
            artifact_id = relative_under_namespace.parts[0] if relative_under_namespace.parts else ""
            if mode == "classifier":
                if artifact_id not in artifact_ids:
                    continue
                # A Maven component is never publishable without its POM. Keep
                # POMs with every classifier-owned artifact even when optional
                # sources, javadocs, and Gradle module metadata are omitted.
                is_optional_metadata = path.name.endswith(
                    ("-sources.jar", "-javadoc.jar", ".module")
                )
                if is_optional_metadata and not include_metadata:
                    continue
                is_metadata = path.suffix == ".pom" or is_optional_metadata
                is_unclassified_runtime = (
                    artifact_id in unclassified_artifact_ids
                    and path.name == f"{artifact_id}-{path.parent.name}.jar"
                )
                if (not is_metadata and not is_unclassified_runtime and classifier_tokens
                        and not any(token in path.name for token in classifier_tokens)):
                    continue
            destination = output / namespace / relative_under_namespace
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)


def shared_native_family(shard: dict, variant: dict) -> str:
    build = shard["build"]
    if build.get("zludaVersion"):
        return "windows-zluda" if shard["os"] == "windows" else "zluda"
    if variant.get("name") == "compat":
        return "compat"
    if build["backend"] in {"vulkan", "hexagon", "tpu"}:
        if build["backend"] == "vulkan" and build.get("javacppPlatform", "").startswith("android-"):
            return f"{build['javacppPlatform']}-vulkan"
        if build["backend"] == "vulkan" and shard["os"] == "windows":
            return "windows-vulkan"
        # A Vulkan base classifier also carries MLIR/Triton support.  The
        # classifier extension—not the presence of MLIR—is what distinguishes
        # the compile-only Vulkan lane from the base runtime lane.
        return (
            "vulkan-mlir"
            if build["backend"] == "vulkan"
            and variant_artifact_classifier(build, variant).endswith("-compile")
            else build["backend"]
        )
    if build["backend"] == "cuda":
        return "windows-cuda" if shard["os"] == "windows" else "linux-cuda"
    return {
        "linux-arm64": "linux-arm64", "windows-x86_64": "windows-cpu",
        "macosx-arm64": "macos-arm64", "android-arm64": "android-arm64",
        "android-x86_64": "android-x86_64", "linux-x86_64": "linux-x86_64",
    }[build["javacppPlatform"]]


def android_api_level(variant: dict) -> int:
    configured = variant.get("androidApi")
    if configured is not None:
        return int(configured)
    return 27 if "nnapi" in variant.get("name", "") else 21


def android_cmake_args(source: Path, build: dict, variant: dict, env: dict[str, str]) -> str:
    if not env.get("ANDROID_NDK") or not env.get("OPENBLAS_PATH"):
        raise ValueError("Android builds require ANDROID_NDK and OPENBLAS_PATH")
    arm64 = build["javacppPlatform"] == "android-arm64"
    abi, toolchain = ("arm64-v8a", "android-arm64.cmake") if arm64 else ("x86_64", "android-x86_64.cmake")
    api = android_api_level(variant)
    return " ".join([
        f"-DCMAKE_TOOLCHAIN_FILE={source / 'libnd4j/cmake' / toolchain}", "-G Ninja",
        "-DSD_ANDROID_BUILD=true", "-DSD_BUILD_WITH_JAVA=OFF",
        f"-DANDROID_ABI={abi}", f"-DANDROID_PLATFORM=android-{api}",
        f"-DANDROID_NDK={env['ANDROID_NDK']}", "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_MAKE_PROGRAM=/usr/bin/ninja",
        f"-DBLAS_LIBRARIES={Path(env['OPENBLAS_PATH']) / 'libopenblas.so'}",
        f"-DLAPACK_LIBRARIES={Path(env['OPENBLAS_PATH']) / 'libopenblas.so'}",
    ])


def shared_variant_helper(variant: dict) -> str:
    """Translate a release-plan variant to the workflow matrix helper verbatim."""
    name = variant.get("name", "")
    if name == "mps-compile":
        return "mps-compile"
    if name == "compile-nnapi":
        return "compile-nnapi"
    if name == "compile":
        # Windows uses the compile classifier for a native MSVC build.  Keep the
        # generic compile helper (which enables managed LLVM/MLIR) for other OSes.
        return "" if variant.get("windowsNativeCompile") else "compile"
    if variant.get("mlir"):
        return "compile"
    return variant.get("helper", "")


def zluda_target(build: dict) -> str:
    arguments = [
        argument.split("=", 1)[1]
        for argument in build.get("mavenArgs", [])
        if argument.startswith("-Dlibnd4j.zluda=")
    ]
    if not build.get("zludaVersion"):
        if arguments:
            raise ValueError("libnd4j.zluda is set without zludaVersion")
        return ""
    if len(arguments) != 1:
        raise ValueError("ZLUDA releases require exactly one -Dlibnd4j.zluda target")
    if arguments[0] != ZLUDA_TARGET:
        raise ValueError(f"unsupported ZLUDA target {arguments[0]!r}; expected {ZLUDA_TARGET}")
    return arguments[0]


def zluda_rocm_classifier_suffix(build: dict) -> str:
    """Return the ROCm-qualified ZLUDA suffix used by native coordinates."""
    if not build.get("zludaVersion"):
        return "-zluda"
    rocm_version = str(build.get("rocmVersion", "")).strip()
    if not rocm_version:
        raise RuntimeError(
            "ZLUDA releases must declare rocmVersion so native classifiers are unambiguous"
        )
    return f"-zluda-rocm-{rocm_version}"


def attest_zluda_configuration(build: dict) -> None:
    version = build.get("zludaVersion")
    if not version:
        return
    target = zluda_target(build)
    failures = []
    if build.get("backend") != "cuda":
        failures.append("backend must be cuda")
    if "zluda" not in build.get("profiles", []):
        failures.append("zluda profile is missing")
    zluda_artifact_id = f"nd4j-zluda-{build.get('cudaVersion', '')}"
    if f":{zluda_artifact_id}" not in build.get("modules", []):
        failures.append(f":{zluda_artifact_id} module is missing")
    native_cuda_artifact_id = f":nd4j-cuda-{build.get('cudaVersion', '')}"
    if native_cuda_artifact_id in build.get("modules", []):
        failures.append(
            f"{native_cuda_artifact_id} must not mediate the ZLUDA classifier"
        )
    variants = build.get("variants", [])
    try:
        rocm_suffix = zluda_rocm_classifier_suffix(build)
    except RuntimeError as error:
        failures.append(str(error))
        rocm_suffix = "-zluda"
    expected_classifier_suffix = f"-cuda-{build.get('cudaVersion', '')}{rocm_suffix}"
    if not variants or any(
            variant.get("classifierSuffix") != expected_classifier_suffix
            or variant.get("platformExtension") != rocm_suffix
            for variant in variants):
        failures.append("ZLUDA classifier/platform extension is not ROCm-qualified")
    if failures:
        raise RuntimeError("ZLUDA configuration attestation failed: " + "; ".join(failures))
    print(
        f"[dl4j-attestation] zludaVersion={version} target={target} "
        f"profile=zluda module=:{zluda_artifact_id} dependencyOwner=cmake ",
        flush=True,
    )


def write_build_result(path: Path | None, completed_variants: list[str]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"schemaVersion": 1, "completedVariants": completed_variants},
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )


def write_build_benchmark(path: Path | None, benchmark: dict) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(benchmark, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_native_platform(source: Path, shard: dict, repository: Path, env: dict[str, str],
                          compiler_cache: str | None, release_version: str | None = None,
                          config: dict | None = None, maven_output: Path | None = None,
                          progress_output: Path | None = None,
                          benchmark: dict | None = None,
                          benchmark_output: Path | None = None) -> None:
    """Invoke the exact shared scripts used by each GitHub platform workflow."""
    build, shard_id = shard["build"], shard["id"]
    rules = shard.get("artifactRules", {})
    enable_sdx_release_component(build, rules)
    reset_unclassified_artifacts(repository, build, rules, release_version)
    prepare_openblas(source, build, env, config)
    prebuilt_libnd4j_url = prepare_prebuilt_libnd4j(source, build, env)
    completed_variants: list[str] = []
    write_build_result(progress_output, completed_variants)
    for variant in build["variants"]:
        print(f"[dl4j-phase] shard={shard_id} phase=native variant={variant['name']}", flush=True)
        reset_variant_classifier_artifacts(
            repository, build, rules, variant, release_version
        )
        variant_env = env.copy()
        family = shared_native_family(shard, variant)
        sdx_library, sdx_links, sdx_output = sdx_native_configuration(
            source, build, variant
        )
        variant_env.update({
            "DL4J_FAMILY": family,
            "DL4J_HELPER": shared_variant_helper(variant),
            "DL4J_EXTENSION": variant.get("extension", ""),
            "DL4J_PLATFORM_EXTENSION": variant_platform_extension(variant),
            "DL4J_CLASSIFIER": variant_libnd4j_classifier(build, variant),
            "DL4J_BUILD_THREADS": str(build.get("buildThreads", 16)),
            "DL4J_MVN_FLAGS": str(build.get("workflowMvnFlags", "")),
            "DL4J_MAVEN_GOAL": "install",
            "DL4J_MAVEN_REPOSITORY": str(repository),
            "DL4J_CUDA_VERSION": str(build.get("cudaVersion", "")),
            "DL4J_ROCM_VERSION": str(build.get("rocmVersion", "")),
            "DL4J_ZLUDA_VERSION": str(build.get("zludaVersion", "")),
            "DL4J_ZLUDA_TARGET": zluda_target(build),
            "DL4J_LIBND4J_URL": prebuilt_libnd4j_url,
            "DL4J_BUILD_SDX": "1" if sdx_enabled_for_build(build) else "0",
            "DL4J_SDX_NATIVE_LIBRARY": sdx_library,
            "DL4J_SDX_PLATFORM_LINKS": sdx_links,
            "DL4J_SDX_OUTPUT_PATH": str(sdx_output),
            "DL4J_SDX_CLASSIFIER": sdx_variant_artifact_classifier(build, variant),
        })
        if family == "vulkan-mlir" and variant.get("mlir"):
            # native-platform.sh uses platform.classifier for the JavaCPP
            # platform, but the compile-only Vulkan/MLIR lane also needs the
            # matching platform extension and libnd4j classifier.  Inject
            # these through the shared Maven flags so the Azure worker can
            # apply the fix without requiring a pushed source commit.
            existing_mvn_flags = variant_env.get("DL4J_MVN_FLAGS", "").strip()
            compile_flags = (
                "-Djavacpp.platform.extension=-compile "
                "-Dlibnd4j.classifier=linux-x86_64-compile"
            )
            variant_env["DL4J_MVN_FLAGS"] = (
                f"{existing_mvn_flags} {compile_flags}".strip()
            )
        if build["javacppPlatform"].startswith("android-"):
            variant_env["DL4J_ANDROID_API"] = str(android_api_level(variant))
            variant_env["DL4J_CMAKE_ARGS"] = android_cmake_args(source, build, variant, variant_env)
        script_name = "linux-x86_64.sh" if family == "linux-x86_64" else "native-platform.sh"
        if family == "linux-x86_64":
            variant_env["DL4J_MATRIX_MVN_EXT"] = variant_env.pop("DL4J_MVN_FLAGS")
            variant_env["DL4J_LIBND4J_FILE_DOWNLOAD"] = prebuilt_libnd4j_url
        if variant.get("mlir") and config is not None:
            restore_remote_dependency_cache(source, config, variant_env)
        variant_started_at = int(time.time())
        variant_timer = time.monotonic()
        variant_status = "failed"
        try:
            run(
                bash_command(
                    [str(source / "build-scripts/release" / script_name), "--run"],
                    variant_env,
                ),
                source,
                variant_env,
            )
            variant_status = "complete"
        finally:
            if benchmark is not None:
                benchmark.setdefault("variants", []).append({
                    "name": variant["name"],
                    "status": variant_status,
                    "startedAt": variant_started_at,
                    "completedAt": int(time.time()),
                    "durationSeconds": round(time.monotonic() - variant_timer, 3),
                })
                write_build_benchmark(benchmark_output, benchmark)
        attest_variant_classifier_artifacts(
            repository, build, rules, variant, release_version, "local-repository"
        )
        completed_variants.append(variant["name"])
        if maven_output is not None:
            maven_output.mkdir(parents=True, exist_ok=True)
            stage_repository(repository, maven_output, rules)
            attest_variant_classifier_artifacts(
                maven_output,
                build,
                rules,
                variant,
                release_version,
                "incremental-staged-repository",
            )
        write_build_result(progress_output, completed_variants)
        if compiler_cache:
            run([compiler_cache, "--show-stats"], source, env)
    attest_unclassified_artifacts(
        repository, build, rules, release_version, "local-repository"
    )
    if build.get("buildCrossPlatform") and has_base_platform_variant(build):
        print(f"[dl4j-phase] shard={shard_id} phase=cross-platform", flush=True)
        build_cross_platform(source, build, repository, env)
    elif build.get("buildCrossPlatform"):
        print(
            f"[dl4j-phase] shard={shard_id} phase=cross-platform status=skipped "
            "reason=no-base-platform-variant",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--maven-output", type=Path, required=True)
    parser.add_argument("--sdk-output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    shard = config["shard"]
    build = shard["build"]
    env = os.environ.copy()
    env["MAVEN_OPTS"] = f"-Xmx{build.get('mavenHeapGiB', 16)}g -Dmaven.repo.local={args.repository}"
    env["DL4J_BUILD_SDX"] = "1" if sdx_enabled_for_build(build) else "0"
    sdx_backend = build.get("backend")
    sdx_library = {
        "cuda": "nd4jcuda",
        "vulkan": "nd4jvulkan",
    }.get(sdx_backend, "nd4jcpu")
    sdx_output_dir = "vulkan" if sdx_backend == "vulkan" else (
        "cuda" if sdx_backend == "cuda" else "cpu"
    )
    env["DL4J_SDX_NATIVE_LIBRARY"] = sdx_library
    env["DL4J_SDX_PLATFORM_LINKS"] = sdx_library
    env["DL4J_SDX_OUTPUT_PATH"] = str(
        args.source / "libnd4j/blasbuild" / sdx_output_dir
    )
    driver_started_at = int(time.time())
    driver_timer = time.monotonic()
    compiler_cache, sccache_started = configure_compiler_cache(
        config, args.source, env
    )
    benchmark_path = args.maven_output.parent / "build-benchmark.json"
    selected_machine = config.get("selectedMachine") or {}
    benchmark = {
        "schemaVersion": 1,
        "shard": shard["id"],
        "provider": config.get("provider"),
        "runId": config.get("runId"),
        "commit": config.get("commit"),
        "machine": {
            "name": selected_machine.get("name"),
            "vcpus": selected_machine.get("vcpus"),
            "memoryGiB": selected_machine.get("memoryGiB"),
        },
        "buildThreads": int(build.get("buildThreads", 16)),
        "mavenHeapGiB": int(build.get("mavenHeapGiB", 16)),
        "compilerCacheBackend": (config.get("compilerCache") or {}).get("backend"),
        "compilerCacheSnapshot": compiler_cache_snapshot_metrics(env),
        "startedAt": driver_started_at,
        "variants": [],
    }
    write_build_benchmark(benchmark_path, benchmark)
    build_completed = False
    try:
        print(f"[dl4j-phase] shard={shard['id']} phase=version-setup", flush=True)
        update = bash_command(
            ["./update-versions.sh", config["snapshotVersion"], config["releaseVersion"]],
            env,
        )
        run(update, args.source, env)
        if build["backend"] == "cuda":
            run(
                bash_command(["./change-cuda-versions.sh", build["cudaVersion"]], env),
                args.source,
                env,
            )
        prepare_rocm_build_toolchain(build, env, config)
        attest_zluda_configuration(build)
        if build.get("kind") == "cross-platform":
            print(f"[dl4j-phase] shard={shard['id']} phase=cross-platform", flush=True)
            build_cross_platform(args.source, build, args.repository, env)
        else:
            build_native_platform(
                args.source,
                shard,
                args.repository,
                env,
                compiler_cache,
                config["releaseVersion"],
                config,
                args.maven_output,
                args.maven_output.parent / "build-result.json",
                benchmark,
                benchmark_path,
            )
        print(f"[dl4j-phase] shard={shard['id']} phase=package", flush=True)
        args.maven_output.mkdir(parents=True, exist_ok=True)
        args.sdk_output.mkdir(parents=True, exist_ok=True)
        rules = shard.get("artifactRules", {})
        stage_repository(args.repository, args.maven_output, rules)
        for variant in build.get("variants", []):
            attest_variant_classifier_artifacts(
                args.maven_output,
                build,
                rules,
                variant,
                config["releaseVersion"],
                "staged-repository",
            )
        attest_unclassified_artifacts(
            args.maven_output,
            build,
            rules,
            config["releaseVersion"],
            "staged-repository",
        )
        write_build_result(
            args.maven_output.parent / "build-result.json",
            [variant["name"] for variant in build.get("variants", [])],
        )
        runtime_count = package_runtime_sdk(
            args.source, args.sdk_output, int(build.get("buildThreads", 16))
        )
        jar_count = package_sdk_jars(
            args.repository, args.sdk_output, build, shard.get("artifactRules", {})
        )
        build_aot(args.source, args.sdk_output, build, args.repository, env)
        if "maven" in shard["workloads"] and not any(
            path.is_file() for path in args.maven_output.rglob("*")
        ):
            raise RuntimeError("Maven workload produced no owned release artifacts")
        if "sdk" in shard["workloads"] and runtime_count == 0:
            raise RuntimeError("SDK workload produced no SDX runtime assets")
        if "sdk" in shard["workloads"] and jar_count == 0:
            raise RuntimeError("SDK workload produced no platform SDK JARs")
        print(f"[dl4j-phase] shard={shard['id']} phase=complete", flush=True)
        build_completed = True
    finally:
        try:
            if sccache_started and compiler_cache:
                print("+", subprocess.list2cmdline([compiler_cache, "--stop-server"]), flush=True)
                stopped = subprocess.run(
                    [compiler_cache, "--stop-server"], cwd=args.source, env=env, check=False,
                )
                if build_completed and stopped.returncode != 0:
                    raise RuntimeError(
                        f"sccache server shutdown failed with exit code {stopped.returncode}"
                    )
            if build_completed:
                benchmark["compilerCacheSnapshot"] = publish_compiler_cache_snapshot(
                    config, env
                )
        finally:
            benchmark.update({
                "status": "complete" if build_completed else "failed",
                "completedAt": int(time.time()),
                "durationSeconds": round(time.monotonic() - driver_timer, 3),
            })
            write_build_benchmark(benchmark_path, benchmark)


if __name__ == "__main__":
    main()
