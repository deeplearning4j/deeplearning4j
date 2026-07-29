#!/usr/bin/env python3
"""Run one platform's existing release matrix outside GitHub Actions."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tarfile
import urllib.request
import zipfile
from pathlib import Path


def run(command: list[str], cwd: Path, env: dict[str, str]) -> None:
    print("+", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


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


def maven_command(build: dict, variant: dict, repository: Path, source: Path | None = None, env: dict[str, str] | None = None) -> list[str]:
    profiles = list(build.get("profiles", []))
    if "sdx" not in profiles:
        profiles.append("sdx")
    command = [
        "mvn", "--batch-mode", "--no-transfer-progress", f"-P{','.join(profiles)}",
        f"-Dmaven.repo.local={repository}", "-DskipTests", "-DskipTestResourceEnforcement=true",
        "-Dmaven.javadoc.failOnError=false", "-Dlibnd4j.generate.flatc=ON",
        "-Dlibnd4j.sdx.standalone=ON", "-Dlibnd4j.sdx.package.runtime=ON",
        "-Dlibnd4j.log=libnd4j-build.log", "-Dlibnd4j.oom.memory.threshold=95",
        "-Dlibnd4j.oom.velocity.threshold=40", "-Dhttp.keepAlive=false",
        "-Dmaven.wagon.http.pool=false", "-Dmaven.wagon.http.retryHandler.count=3",
        f"-Dlibnd4j.buildthreads={build.get('buildThreads', 16)}",
        f"-Djavacpp.platform={build['javacppPlatform']}",
    ]
    command.extend(build.get("mavenArgs", []))
    if build["javacppPlatform"].startswith("android-"):
        if source is None or env is None or not env.get("ANDROID_NDK") or not env.get("OPENBLAS_PATH"):
            raise ValueError("Android builds require source, ANDROID_NDK and OPENBLAS_PATH")
        arm64 = build["javacppPlatform"] == "android-arm64"
        abi = "arm64-v8a" if arm64 else "x86_64"
        toolchain = "android-arm64.cmake" if arm64 else "android-x86_64.cmake"
        api = 27 if "nnapi" in variant.get("name", "") else 21
        compiler = Path(env["ANDROID_NDK"]) / "toolchains/llvm/prebuilt/linux-x86_64/bin/clang++"
        cmake_args = " ".join([
            f"-DCMAKE_TOOLCHAIN_FILE={source / 'libnd4j/cmake' / toolchain}", "-G Ninja",
            "-DSD_ANDROID_BUILD=true", f"-DANDROID_ABI={abi}", f"-DANDROID_PLATFORM=android-{api}",
            f"-DANDROID_NDK={env['ANDROID_NDK']}", "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_MAKE_PROGRAM=/usr/bin/ninja",
            f"-DBLAS_LIBRARIES={Path(env['OPENBLAS_PATH']) / 'libopenblas.so'}",
            f"-DLAPACK_LIBRARIES={Path(env['OPENBLAS_PATH']) / 'libopenblas.so'}",
        ])
        command.extend([f"-Djavacpp.platform.compiler={compiler}", f"-Dlibnd4j.cmake={cmake_args}"])
    command.extend(variant_flags(build, variant))
    command.extend(["-pl", ",".join(build["modules"]), "--also-make", "install"])
    return command


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


def build_native_platform(source: Path, build: dict, repository: Path, env: dict[str, str],
                          compiler_cache: str | None, shard_id: str) -> None:
    """Build local native dependencies before the GitHub-equivalent Java stages."""
    prepare_openblas(source, build, env)
    for variant in build["variants"]:
        print(f"[dl4j-phase] shard={shard_id} phase=native variant={variant['name']}", flush=True)
        variant_env = env.copy()
        if build["backend"] == "cpu" and build["javacppPlatform"] == "linux-x86_64":
            variant_env.update({
                "DL4J_HELPER": "compile" if variant.get("mlir") else variant.get("helper", ""),
                "DL4J_EXTENSION": variant.get("extension", ""),
                "DL4J_LIBND4J_FILE_DOWNLOAD": "",
                "DL4J_BUILD_THREADS": str(build.get("buildThreads", 16)),
                "DL4J_MATRIX_MVN_EXT": " ".join(build.get("mavenArgs", [])),
                "DL4J_MAVEN_GOAL": "install",
                "DL4J_MAVEN_REPOSITORY": str(repository),
            })
            command = ["bash", str(source / "build-scripts/release/linux-x86_64.sh"), "--run"]
        else:
            command = maven_command(build, variant, repository, source, variant_env)
        run(command, source, variant_env)
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
    compiler_cache = "sccache" if shutil.which("sccache") else ("ccache" if shutil.which("ccache") else None)
    if compiler_cache:
        cache_dir = args.source.parent / compiler_cache
        cache_dir.mkdir(parents=True, exist_ok=True)
        env.update({
            "CMAKE_C_COMPILER_LAUNCHER": compiler_cache,
            "CMAKE_CXX_COMPILER_LAUNCHER": compiler_cache,
            "CMAKE_CUDA_COMPILER_LAUNCHER": compiler_cache,
        })
        if compiler_cache == "ccache":
            env.update({"CCACHE_DIR": str(cache_dir), "CCACHE_BASEDIR": str(args.source),
                        "CCACHE_NOHASHDIR": "true", "CCACHE_MAXSIZE": "100G"})
            run([compiler_cache, "--zero-stats"], args.source, env)
        else:
            env.update({"SCCACHE_DIR": str(cache_dir), "SCCACHE_CACHE_SIZE": "100G",
                        "SCCACHE_IDLE_TIMEOUT": "0"})
            run([compiler_cache, "--start-server"], args.source, env)
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
        build_native_platform(args.source, build, args.repository, env, compiler_cache, shard["id"])
    print(f"[dl4j-phase] shard={shard['id']} phase=package", flush=True)
    args.maven_output.mkdir(parents=True, exist_ok=True)
    args.sdk_output.mkdir(parents=True, exist_ok=True)
    stage_repository(args.repository, args.maven_output, shard.get("artifactRules", {}))
    runtime_count = package_runtime_sdk(args.source, args.sdk_output, int(build.get("buildThreads", 16)))
    jar_count = package_sdk_jars(args.repository, args.sdk_output, build, shard.get("artifactRules", {}))
    build_aot(args.source, args.sdk_output, build, args.repository, env)
    if "maven" in shard["workloads"] and not any(path.is_file() for path in args.maven_output.rglob("*")):
        raise RuntimeError("Maven workload produced no owned release artifacts")
    if "sdk" in shard["workloads"] and runtime_count == 0:
        raise RuntimeError("SDK workload produced no SDX runtime assets")
    if "sdk" in shard["workloads"] and jar_count == 0:
        raise RuntimeError("SDK workload produced no platform SDK JARs")
    print(f"[dl4j-phase] shard={shard['id']} phase=complete", flush=True)


if __name__ == "__main__":
    main()
