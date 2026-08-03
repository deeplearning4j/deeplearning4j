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
import urllib.request
import zipfile
from pathlib import Path


SCCACHE_VERSION = "0.15.0"
SCCACHE_RELEASE_BASE = (
    f"https://github.com/mozilla/sccache/releases/download/v{SCCACHE_VERSION}"
)
SCCACHE_ASSETS = {
    ("linux", "x86_64"): (
        "x86_64-unknown-linux-musl",
        "782d2b5dd7ae0a55ebe368ab258114d0928d019ac2d949ab85d5d02f3926709e",
    ),
    ("linux", "arm64"): (
        "aarch64-unknown-linux-musl",
        "3a6a3712b49da3d263bf2d30d702de4302793016019e800bfb81c0c69401d8f8",
    ),
    ("macos", "arm64"): (
        "aarch64-apple-darwin",
        "430ef7b5f54256d3ed5bfe77e8b0afc51aa209aeebe4f95b69c3a52ce3acc6e9",
    ),
    ("windows", "x86_64"): (
        "x86_64-pc-windows-msvc",
        "b0b257a164bf438b2dea134ca7ded41c100f59a64b3bf275a202f1e8102ab217",
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


def prepare_zluda(source: Path, build: dict, env: dict[str, str]) -> None:
    version = build.get("zludaVersion")
    if not version:
        return
    request = urllib.request.Request(
        f"https://api.github.com/repos/vosen/ZLUDA/releases/tags/{version}",
        headers={"Accept": "application/vnd.github+json", "User-Agent": "dl4j-release-builder"},
    )
    with urllib.request.urlopen(request) as response:
        release = json.load(response)
    assets = [asset for asset in release.get("assets", []) if "linux" in asset["name"].lower()]
    if len(assets) != 1:
        raise RuntimeError(f"expected one Linux ZLUDA asset for {version}, found {[a['name'] for a in assets]}")
    archive = source / assets[0]["name"]
    urllib.request.urlretrieve(assets[0]["browser_download_url"], archive)
    target = source / "zluda"
    target.mkdir()
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(target)
    else:
        with tarfile.open(archive) as bundle:
            bundle.extractall(target)
    libraries = list(target.rglob("libcuda.so")) + list(target.rglob("libcuda.so.*"))
    if not libraries:
        raise RuntimeError(f"ZLUDA release {version} contains no libcuda.so")
    library = libraries[0]
    root = library.parent.parent if library.parent.name == "lib" else library.parent
    env["ZLUDA_PATH"] = str(root)


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
    classifiers = set()
    for variant in build["variants"]:
        classifier = next(flag.split("=", 1)[1] for flag in variant_flags(build, variant)
                          if flag.startswith("-Dlibnd4j.classifier="))
        classifiers.add(classifier)
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
                if not is_metadata and classifier_tokens and not any(token in path.name for token in classifier_tokens):
                    continue
            destination = output / namespace / relative_under_namespace
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)


def shared_native_family(shard: dict, variant: dict) -> str:
    build = shard["build"]
    if build.get("zludaVersion"):
        return "zluda"
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


def android_cmake_args(source: Path, build: dict, variant: dict, env: dict[str, str]) -> str:
    if not env.get("ANDROID_NDK") or not env.get("OPENBLAS_PATH"):
        raise ValueError("Android builds require ANDROID_NDK and OPENBLAS_PATH")
    arm64 = build["javacppPlatform"] == "android-arm64"
    abi, toolchain = ("arm64-v8a", "android-arm64.cmake") if arm64 else ("x86_64", "android-x86_64.cmake")
    api = 27 if "nnapi" in variant.get("name", "") else 21
    return " ".join([
        f"-DCMAKE_TOOLCHAIN_FILE={source / 'libnd4j/cmake' / toolchain}", "-G Ninja",
        "-DSD_ANDROID_BUILD=true", f"-DANDROID_ABI={abi}", f"-DANDROID_PLATFORM=android-{api}",
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
    for argument in build.get("mavenArgs", []):
        if argument.startswith("-Dlibnd4j.zluda="):
            return argument.split("=", 1)[1]
    return "rocm6"


def build_native_platform(source: Path, shard: dict, repository: Path, env: dict[str, str],
                          compiler_cache: str | None) -> None:
    """Invoke the exact shared scripts used by each GitHub platform workflow."""
    build, shard_id = shard["build"], shard["id"]
    prepare_openblas(source, build, env)
    for variant in build["variants"]:
        print(f"[dl4j-phase] shard={shard_id} phase=native variant={variant['name']}", flush=True)
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
            variant_env["DL4J_CMAKE_ARGS"] = android_cmake_args(source, build, variant, variant_env)
        script_name = "linux-x86_64.sh" if family == "linux-x86_64" else "native-platform.sh"
        if family == "linux-x86_64":
            variant_env["DL4J_MATRIX_MVN_EXT"] = variant_env.pop("DL4J_MVN_FLAGS")
            variant_env["DL4J_LIBND4J_FILE_DOWNLOAD"] = ""
        run(["bash", str(source / "build-scripts/release" / script_name), "--run"], source, variant_env)
        if compiler_cache:
            run([compiler_cache, "--show-stats"], source, env)
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
        prepare_zluda(args.source, build, env)
        if build.get("kind") == "cross-platform":
            print(f"[dl4j-phase] shard={shard['id']} phase=cross-platform", flush=True)
            build_cross_platform(args.source, build, args.repository, env)
        else:
            build_native_platform(args.source, shard, args.repository, env, compiler_cache)
        print(f"[dl4j-phase] shard={shard['id']} phase=package", flush=True)
        args.maven_output.mkdir(parents=True, exist_ok=True)
        args.sdk_output.mkdir(parents=True, exist_ok=True)
        stage_repository(args.repository, args.maven_output, shard.get("artifactRules", {}))
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
