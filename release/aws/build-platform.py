#!/usr/bin/env python3
"""Run one platform's existing release matrix outside GitHub Actions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
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
ZLUDA_WINDOWS_REQUIRED_FILES = (
    "nvcuda.dll",
    "nvcudart_hybrid64.dll",
    "zluda.exe",
    "zluda_redirect.dll",
)
ZLUDA_ASSETS = {
    ("v6", "linux"): (
        "zluda-linux-3fe12063.tar.gz",
        "d9fd9893abaf3206c56d3eb25f0475c6327aa8de8e77f21be8a24f275556c3e1",
    ),
    ("v6", "windows"): (
        "zluda-windows-3fe1206.zip",
        "fda8891c6fdfaba438f2eb0f9d749ffa2c1fddbdf225be2301f0d7a25e37208a",
    ),
}
ROCM_BUILD_SDKS = {
    "7.2.4": {
        "installer_name": "amdgpu-install_7.2.4.70204-1_all.deb",
        "installer_url": (
            "https://repo.radeon.com/amdgpu-install/7.2.4/ubuntu/jammy/"
            "amdgpu-install_7.2.4.70204-1_all.deb"
        ),
        "component_packages": {
            "hip": ("rocm-hip-runtime-dev",),
            "miopen": ("miopen-hip-dev",),
        },
    },
}
ROCM_BUILD_COMPONENTS = ("hip", "miopen")
DOWNLOAD_RETRIES = 4
TRANSIENT_HTTP_STATUSES = {408, 429, 500, 502, 503, 504}


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


def _required_cache_value(settings: dict, name: str) -> str:
    value = settings.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"compilerCache.{name} must be a non-empty string")
    return value.strip()


def _configure_compiler_launchers(env: dict[str, str], compiler_cache: str) -> None:
    env.update({
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
        cache_dir = source.parent / "sccache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        compiler_cache = ensure_sccache(cache_dir)
        env.update({
            "SCCACHE_DIR": str(cache_dir),
            "SCCACHE_CACHE_SIZE": "100G",
            "SCCACHE_IDLE_TIMEOUT": "0",
            "SCCACHE_BASEDIRS": str(source.resolve()),
            "SCCACHE_ERROR_LOG": str(cache_dir / "sccache-error.log"),
            "SCCACHE_MULTILEVEL_CHAIN": f"disk,{backend}",
            "SCCACHE_MULTILEVEL_WRITE_ERROR_POLICY": "all",
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
            env.update({
                "SCCACHE_AZURE_CONNECTION_STRING": _required_cache_value(
                    remote, "connectionString"
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


def variant_flags(build: dict, variant: dict) -> list[str]:
    platform = build["javacppPlatform"]
    backend = build["backend"]
    helper = variant.get("helper", "")
    extension = variant.get("extension", "")
    suffix = variant.get("suffix", "")
    classifier_suffix = variant.get("classifierSuffix", suffix)
    platform_extension = variant.get("platformExtension", suffix)
    flags = [f"-Dlibnd4j.classifier={platform}{classifier_suffix}"]
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
    platform_extension = variant.get(
        "platformExtension", variant.get("suffix", "")
    )
    return f"{build['javacppPlatform']}{platform_extension}"


def required_classifier_artifact_ids(build: dict, rules: dict) -> tuple[str, ...]:
    """Return the CPU/CUDA artifacts that must carry every planned classifier."""
    if rules.get("mode", "all") != "classifier":
        return ()
    backend = build.get("backend")
    if backend == "cpu":
        primary = "nd4j-native"
    elif backend == "cuda":
        cuda_version = str(build.get("cudaVersion", ""))
        if not cuda_version:
            raise ValueError("CUDA classifier validation requires cudaVersion")
        primary = f"nd4j-cuda-{cuda_version}"
    else:
        return ()
    required = (primary, f"{primary}-preset")
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
    classifier = variant_artifact_classifier(build, variant)
    for artifact_id in required:
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
    classifier = variant_artifact_classifier(build, variant)
    found: dict[str, list[Path]] = {}
    for artifact_id in required:
        found[artifact_id] = [
            path
            for path in exact_classifier_jar_candidates(
                repository, artifact_id, version, classifier
            )
            if path.is_file()
        ]
    missing = [artifact_id for artifact_id, paths in found.items() if not paths]
    if missing:
        raise RuntimeError(
            f"{phase} is missing exact {classifier} classifier JARs for "
            f"variant {variant['name']}: {', '.join(missing)}"
        )
    paths = sorted(
        path.relative_to(repository).as_posix()
        for matches in found.values()
        for path in matches
    )
    print(
        f"[dl4j-attestation] phase={phase} variant={variant['name']} "
        f"classifier={classifier} artifacts={','.join(paths)}",
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
    run(["bash", str(script), "--run-tokenizers"], source, cross_env)
    run(["bash", str(script), "--run-java"], source, cross_env)


def prepare_openblas(source: Path, build: dict, env: dict[str, str]) -> None:
    if ":libnd4j" not in build.get("modules", []):
        return
    classifier = build["javacppPlatform"]
    version = "0.3.28-1.5.11"
    archive = source / f"openblas-{version}-{classifier}.jar"
    url = f"https://repo1.maven.org/maven2/org/bytedeco/openblas/{version}/{archive.name}"
    print(f"+ download {url}", flush=True)
    urllib.request.urlretrieve(url, archive)
    target = source / "openblas_home"
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(target)
    headers = list(target.rglob("include/cblas.h"))
    if not headers:
        raise RuntimeError(f"OpenBLAS archive has no include/cblas.h: {archive}")
    env["OPENBLAS_PATH"] = str(headers[0].parent.parent)


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
    miopen_header = rocm_root / "include/miopen/miopen.h"
    miopen_runtime = _first_existing_file(rocm_root, (
        "lib/libMIOpen.so",
        "lib64/libMIOpen.so",
        "lib/x86_64-linux-gnu/libMIOpen.so",
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
        "miopenHeader": miopen_header,
        "miopenRuntime": miopen_runtime,
    }
    if emit:
        print(
            "[dl4j-attestation] "
            f"rocmVersion={spec['version']} rocmBuildOnly=true "
            f"components={','.join(spec['components'])} root={rocm_root} "
            f"hipHeader={hip_header} hipRuntime={hip_runtime} "
            f"miopenHeader={miopen_header} miopenRuntime={miopen_runtime} "
            "hardwareProbe=skipped",
            flush=True,
        )
    return attested


def prepare_rocm_build_toolchain(build: dict, env: dict[str, str]) -> None:
    """Install the pinned userspace ROCm SDK on a CPU builder; never install a driver."""
    spec = rocm_build_spec(build)
    if spec is None:
        return
    try:
        attest_rocm_build_toolchain(build, env, emit=False)
    except RuntimeError as initial_failure:
        if platform.system().lower() != "linux" or platform.machine().lower() not in {
                "amd64", "x86_64"}:
            raise RuntimeError(
                "ROCm build-only provisioning requires a Linux x86_64 builder"
            ) from initial_failure
        if not hasattr(os, "geteuid") or os.geteuid() != 0:
            raise RuntimeError(
                "ROCm build-only provisioning requires root inside the disposable build container"
            ) from initial_failure
        install_env = env.copy()
        install_env["DEBIAN_FRONTEND"] = "noninteractive"
        with tempfile.TemporaryDirectory(prefix="dl4j-rocm-sdk-") as temporary_directory:
            installer = Path(temporary_directory) / str(spec["installer_name"])
            download_with_retry(
                str(spec["installer_url"]),
                installer,
                f"ROCm {spec['version']} Ubuntu Jammy repository installer",
            )
            run(["apt-get", "update"], Path("/"), install_env)
            run([
                "apt-get", "install", "-y", "--no-install-recommends", str(installer),
            ], Path("/"), install_env)
            run(["apt-get", "update"], Path("/"), install_env)
            run([
                "apt-get", "install", "-y", "--no-install-recommends",
                *spec["packages"],
            ], Path("/"), install_env)
    attest_rocm_build_toolchain(build, env)


def zluda_platform(build: dict) -> str:
    platform = build.get("javacppPlatform", "")
    if platform.startswith("windows-"):
        return "windows"
    if platform.startswith("linux-"):
        return "linux"
    raise ValueError(f"ZLUDA releases do not support JavaCPP platform {platform!r}")


def pinned_zluda_asset(version: str, platform_name: str) -> tuple[str, str]:
    key = (version, platform_name)
    if key not in ZLUDA_ASSETS:
        raise RuntimeError(
            f"ZLUDA {version} has no pinned release asset for {platform_name}"
        )
    return ZLUDA_ASSETS[key]


def find_zluda_runtime(root: Path, build: dict) -> Path | None:
    platform = zluda_platform(build)
    if platform == "windows":
        candidates = [
            candidate
            for candidate in root.rglob("nvcuda.dll")
            if candidate.is_file()
            and all((candidate.parent / name).is_file() for name in ZLUDA_WINDOWS_REQUIRED_FILES)
        ]
    else:
        candidates = [
            candidate
            for candidate in list(root.rglob("libcuda.so")) + list(root.rglob("libcuda.so.*"))
            if candidate.is_file()
        ]
    return (
        min(candidates, key=lambda path: (len(path.relative_to(root).parts), str(path)))
        if candidates else None
    )


def prepare_zluda(source: Path, build: dict, env: dict[str, str]) -> None:
    version = build.get("zludaVersion")
    if not version:
        return
    platform = zluda_platform(build)
    asset_name, expected_digest = pinned_zluda_asset(version, platform)
    archive = source / asset_name
    download_with_retry(
        f"https://github.com/vosen/ZLUDA/releases/download/{version}/{asset_name}",
        archive,
        f"ZLUDA {platform} asset for {version}",
    )
    actual_digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if actual_digest != expected_digest:
        archive.unlink(missing_ok=True)
        raise RuntimeError(
            f"ZLUDA archive SHA-256 mismatch: expected {expected_digest}, got {actual_digest}"
        )
    target = source / "zluda"
    target.mkdir()
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(target)
    else:
        with tarfile.open(archive) as bundle:
            bundle.extractall(target)
    runtime = find_zluda_runtime(target, build)
    if runtime is None:
        expected = "nvcuda.dll" if platform == "windows" else "libcuda.so"
        raise RuntimeError(f"ZLUDA {platform} release {version} contains no {expected}")
    runtime_directory = str(runtime.parent)
    env["ZLUDA_PATH"] = runtime_directory
    search_variable = "PATH" if platform == "windows" else "LD_LIBRARY_PATH"
    current_search_path = env.get(search_variable, "")
    env[search_variable] = (
        runtime_directory + os.pathsep + current_search_path
        if current_search_path else runtime_directory
    )


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
                is_metadata = path.suffix == ".pom" or path.name.endswith(("-sources.jar", "-javadoc.jar", ".module"))
                if is_metadata and not include_metadata:
                    continue
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
        return "vulkan-mlir" if build["backend"] == "vulkan" and variant.get("mlir") else build["backend"]
    if build["backend"] == "cuda":
        return "windows-cuda" if shard["os"] == "windows" else "linux-cuda"
    return {
        "linux-arm64": "linux-arm64", "windows-x86_64": "windows-cpu",
        "macosx-arm64": "macos-arm64", "android-arm64": "android-arm64",
        "android-x86_64": "android-x86_64", "linux-x86_64": "linux-x86_64",
    }[build["javacppPlatform"]]


def android_api_level(variant: dict) -> int:
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
        return "compile"
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


def attest_zluda_configuration(build: dict, env: dict[str, str]) -> None:
    version = build.get("zludaVersion")
    if not version:
        return
    target = zluda_target(build)
    failures = []
    if build.get("backend") != "cuda":
        failures.append("backend must be cuda")
    if "zluda" not in build.get("profiles", []):
        failures.append("zluda profile is missing")
    if ":nd4j-zluda" not in build.get("modules", []):
        failures.append(":nd4j-zluda module is missing")
    variants = build.get("variants", [])
    expected_classifier_suffix = f"-cuda-{build.get('cudaVersion', '')}-zluda"
    if not variants or any(
            variant.get("classifierSuffix") != expected_classifier_suffix
            or variant.get("platformExtension") != "-zluda"
            for variant in variants):
        failures.append("ZLUDA classifier/platform extension is not active")
    zluda_path = Path(env.get("ZLUDA_PATH", ""))
    runtime = None
    if not env.get("ZLUDA_PATH") or not zluda_path.is_dir():
        failures.append("prepared ZLUDA_PATH is missing")
    else:
        runtime = find_zluda_runtime(zluda_path, build)
        if runtime is None:
            failures.append(f"prepared ZLUDA_PATH contains no {zluda_platform(build)} runtime")
    if failures:
        raise RuntimeError("ZLUDA configuration attestation failed: " + "; ".join(failures))
    print(
        f"[dl4j-attestation] zludaVersion={version} target={target} "
        f"profile=zluda module=:nd4j-zluda path={zluda_path} runtime={runtime}",
        flush=True,
    )


def build_native_platform(source: Path, shard: dict, repository: Path, env: dict[str, str],
                          compiler_cache: str | None, release_version: str | None = None) -> None:
    """Invoke the exact shared scripts used by each GitHub platform workflow."""
    build, shard_id = shard["build"], shard["id"]
    rules = shard.get("artifactRules", {})
    reset_unclassified_artifacts(repository, build, rules, release_version)
    prepare_openblas(source, build, env)
    for variant in build["variants"]:
        print(f"[dl4j-phase] shard={shard_id} phase=native variant={variant['name']}", flush=True)
        reset_variant_classifier_artifacts(
            repository, build, rules, variant, release_version
        )
        variant_env = env.copy()
        family = shared_native_family(shard, variant)
        variant_env.update({
            "DL4J_FAMILY": family,
            "DL4J_HELPER": shared_variant_helper(variant),
            "DL4J_EXTENSION": variant.get("extension", ""),
            "DL4J_BUILD_THREADS": str(build.get("buildThreads", 16)),
            "DL4J_MVN_FLAGS": str(build.get("workflowMvnFlags", "")),
            "DL4J_MAVEN_GOAL": "install",
            "DL4J_MAVEN_REPOSITORY": str(repository),
            "DL4J_CUDA_VERSION": str(build.get("cudaVersion", "")),
            "DL4J_ZLUDA_TARGET": zluda_target(build),
        })
        if build["javacppPlatform"].startswith("android-"):
            variant_env["DL4J_ANDROID_API"] = str(android_api_level(variant))
            variant_env["DL4J_CMAKE_ARGS"] = android_cmake_args(source, build, variant, variant_env)
        script_name = "linux-x86_64.sh" if family == "linux-x86_64" else "native-platform.sh"
        if family == "linux-x86_64":
            variant_env["DL4J_MATRIX_MVN_EXT"] = variant_env.pop("DL4J_MVN_FLAGS")
            variant_env["DL4J_LIBND4J_FILE_DOWNLOAD"] = ""
        run(["bash", str(source / "build-scripts/release" / script_name), "--run"], source, variant_env)
        attest_variant_classifier_artifacts(
            repository, build, rules, variant, release_version, "local-repository"
        )
        if compiler_cache:
            run([compiler_cache, "--show-stats"], source, env)
    attest_unclassified_artifacts(
        repository, build, rules, release_version, "local-repository"
    )
    if build.get("buildCrossPlatform"):
        print(f"[dl4j-phase] shard={shard_id} phase=cross-platform", flush=True)
        build_cross_platform(source, build, repository, env)


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
    compiler_cache, sccache_started = configure_compiler_cache(
        config, args.source, env
    )
    build_completed = False
    try:
        print(f"[dl4j-phase] shard={shard['id']} phase=version-setup", flush=True)
        update = ["bash", "./update-versions.sh", config["snapshotVersion"], config["releaseVersion"]]
        run(update, args.source, env)
        if build["backend"] == "cuda":
            run(["bash", "./change-cuda-versions.sh", build["cudaVersion"]], args.source, env)
        prepare_rocm_build_toolchain(build, env)
        prepare_zluda(args.source, build, env)
        attest_zluda_configuration(build, env)
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
        if sccache_started and compiler_cache:
            print("+", subprocess.list2cmdline([compiler_cache, "--stop-server"]), flush=True)
            stopped = subprocess.run(
                [compiler_cache, "--stop-server"], cwd=args.source, env=env, check=False,
            )
            if build_completed and stopped.returncode != 0:
                raise RuntimeError(
                    f"sccache server shutdown failed with exit code {stopped.returncode}"
                )


if __name__ == "__main__":
    main()
