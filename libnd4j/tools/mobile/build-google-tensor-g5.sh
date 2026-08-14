#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIBND4J_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_ROOT="$(cd "$LIBND4J_DIR/.." && pwd)"
PROFILE="$SCRIPT_DIR/profiles/google-tensor-g5.env"
MODULE="$REPO_ROOT/nd4j/nd4j-backends/nd4j-backend-impls/nd4j-sdx-litertlm"
PRESET_MODULE="$REPO_ROOT/nd4j/nd4j-backends/nd4j-backend-impls/nd4j-sdx-preset"
SDX_MODEL_MODULE="$REPO_ROOT/nd4j/nd4j-backends/nd4j-backend-impls/nd4j-sdx-model"
VERIFY_SCRIPT="$SCRIPT_DIR/verify-google-tensor-g5-aar.sh"
BUILD_ENV="$SCRIPT_DIR/android-build-env.sh"
[[ -r "$BUILD_ENV" ]] || {
    echo "Shared Android build discovery is missing: $BUILD_ENV" >&2
    exit 1
}
# shellcheck source=android-build-env.sh
source "$BUILD_ENV"

usage() {
    cat <<'USAGE'
Usage:
  build-google-tensor-g5.sh [options]

Android NDK r28, JDK 17, Maven, the /tmp build root, and bounded host
parallelism are discovered automatically. The build is offline by default.

Options:
  --android-ndk <path> Override NDK discovery
  --java-home <path>   Override JDK 17 discovery
  --maven <command>    Override Maven discovery
  --output-dir <path>  Override the build and distribution root
  --source-dir <path>  Reuse a pinned LiteRT-LM checkout
  --jobs <count>       Parallel jobs (default: min(host CPUs, 8))
  --offline            Forbid network access (default)
  --online             Permit dependency resolution
  --print-config       Print resolved inputs and exit
  --skip-native        Reuse already-built native outputs
  --skip-java          Reuse already-built Java outputs

Builds the direct Google Tensor G5 LiteRT-LM NPU provider and its JavaCPP AAR.
The build is pinned, AOT-only, device-only, and rejects BLAS/host fallback.
No model is downloaded or bundled; target objects are compiled and cached from
canonical .sdz inputs.
USAGE
}

ANDROID_NDK_ARG=""
JAVA_HOME_ARG=""
MAVEN_ARG=""
OUTPUT_DIR=""
SOURCE_DIR=""
JOBS="${JOBS:-$(sdx_android_default_jobs)}"
OFFLINE=1
PRINT_CONFIG=0
SKIP_NATIVE=0
SKIP_JAVA=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --android-ndk)
            ANDROID_NDK_ARG="${2:?missing value for --android-ndk}"
            shift 2
            ;;
        --java-home)
            JAVA_HOME_ARG="${2:?missing value for --java-home}"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="${2:?missing value for --output-dir}"
            shift 2
            ;;
        --source-dir)
            SOURCE_DIR="${2:?missing value for --source-dir}"
            shift 2
            ;;
        --maven)
            MAVEN_ARG="${2:?missing value for --maven}"
            shift 2
            ;;
        --jobs)
            JOBS="${2:?missing value for --jobs}"
            shift 2
            ;;
        --offline)
            OFFLINE=1
            shift
            ;;
        --online)
            OFFLINE=0
            shift
            ;;
        --print-config)
            PRINT_CONFIG=1
            shift
            ;;
        --skip-native)
            SKIP_NATIVE=1
            shift
            ;;
        --skip-java)
            SKIP_JAVA=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

ANDROID_NDK_ARG="$(sdx_android_resolve_ndk "$ANDROID_NDK_ARG")"
JAVA_HOME_REAL="$(sdx_android_resolve_java17 "$JAVA_HOME_ARG")"
MAVEN_CMD="$(sdx_android_resolve_maven "$MAVEN_ARG" "$REPO_ROOT")"
export JAVA_HOME="$JAVA_HOME_REAL"

if [[ ! -f "$PROFILE" ]]; then
    echo "Missing Google Tensor profile: $PROFILE" >&2
    exit 1
fi
# shellcheck disable=SC1090
source "$PROFILE"

if [[ "$SDX_DEVICE_ONLY" != "1" ||
      "$SDX_AOT_ONLY" != "1" ||
      "$SDX_ALLOW_HOST_FALLBACK" != "0" ||
      "$SDX_EXPECT_BLAS" != "0" ||
      "$LITERTLM_BACKEND" != "npu" ||
      "$GOOGLE_TENSOR_SOC" != "Tensor_G5" ]]; then
    echo "Unsafe Google Tensor profile: device/NPU/no-BLAS policy changed" >&2
    exit 1
fi
if [[ -z "$ANDROID_NDK_ARG" || ! -d "$ANDROID_NDK_ARG" ]]; then
    echo "--android-ndk must point to Android NDK r28b or newer" >&2
    exit 2
fi
if [[ ! "$JOBS" =~ ^[1-9][0-9]*$ ]]; then
    echo "--jobs must be a positive integer" >&2
    exit 2
fi

case "$(uname -s)" in
    Linux) NDK_HOST_TAG=linux-x86_64 ;;
    Darwin) NDK_HOST_TAG=darwin-x86_64 ;;
    *)
        echo "Unsupported host: $(uname -s)" >&2
        exit 1
        ;;
esac
READELF="$ANDROID_NDK_ARG/toolchains/llvm/prebuilt/$NDK_HOST_TAG/bin/llvm-readelf"
if [[ ! -x "$READELF" ]]; then
    echo "NDK llvm-readelf not found: $READELF" >&2
    exit 1
fi

for command_name in git npm unzip zip sha256sum; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        echo "Required build command not found: $command_name" >&2
        exit 1
    fi
done
if ! command -v "$MAVEN_CMD" >/dev/null 2>&1 && [[ ! -x "$MAVEN_CMD" ]]; then
    echo "Maven command not found: $MAVEN_CMD" >&2
    exit 1
fi

BUILD_ROOT="${OUTPUT_DIR:-$(sdx_android_default_build_root)/accelerator/google-tensor-g5}"
BUILD_ROOT="$(realpath -m -- "$BUILD_ROOT")"
SOURCE_DIR="${SOURCE_DIR:-$BUILD_ROOT/source/litert-lm-$LITERTLM_VERSION}"
CACHE_DIR="$BUILD_ROOT/cache"
BAZEL_OUTPUT_ROOT="$CACHE_DIR/bazel-google-tensor-g5"
BAZEL_LFS_ROOT="$CACHE_DIR/bazel-git-lfs"
NPM_CACHE="$CACHE_DIR/npm"
DIST_DIR="$BUILD_ROOT/dist"
mkdir -p "$CACHE_DIR" "$DIST_DIR"

printf 'Resolved Google Tensor G5 build configuration:\n'
printf '  Android NDK:  %s\n' "$ANDROID_NDK_ARG"
printf '  JDK 17:       %s\n' "$JAVA_HOME_REAL"
printf '  Maven:        %s\n' "$MAVEN_CMD"
printf '  build root:   %s\n' "$BUILD_ROOT"
printf '  source:       %s\n' "$SOURCE_DIR"
printf '  offline:      %s\n' "$OFFLINE"
printf '  jobs:         %s\n' "$JOBS"
[[ "$PRINT_CONFIG" == 0 ]] || exit 0

export OPENBLAS_NUM_THREADS=0
unset OPENBLAS_HOME MKLROOT BLAS_HOME || true

NPM_FLAGS=(--yes)
MAVEN_FLAGS=()
BAZEL_OFFLINE_FLAGS=()
if [[ "$OFFLINE" == "1" ]]; then
    NPM_FLAGS+=(--offline)
    MAVEN_FLAGS+=(-o)
    BAZEL_OFFLINE_FLAGS+=(--repository_disable_download)
fi

run_bazelisk() {
    XDG_CACHE_HOME="$CACHE_DIR/xdg"     npm_config_cache="$NPM_CACHE"     USE_BAZEL_VERSION="$LITERTLM_BAZEL_VERSION"         npx "${NPM_FLAGS[@]}"         "@bazel/bazelisk@$LITERTLM_BAZELISK_VERSION" "$@"
}

bootstrap_git_lfs() {
    if command -v git-lfs >/dev/null 2>&1; then
        command -v git-lfs
        return
    fi

    if [[ "$OFFLINE" == "1" ]]; then
        echo "Offline build requires git-lfs or a populated Bazel LFS cache" >&2
    fi

    local bootstrap="$CACHE_DIR/git-lfs-bootstrap"
    mkdir -p "$bootstrap"
    cat > "$bootstrap/WORKSPACE.bazel" <<'WORKSPACE'
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "git_lfs",
    urls = [
        "https://github.com/git-lfs/git-lfs/releases/download/v3.7.1/git-lfs-linux-amd64-v3.7.1.tar.gz",
    ],
    sha256 = "1c0b6ee5200ca708c5cebebb18fdeb0e1c98f1af5c1a9cba205a4c0ab5a5ec08",
    strip_prefix = "git-lfs-3.7.1",
    build_file_content = """
exports_files(["git-lfs"])
""",
)
WORKSPACE
    cat > "$bootstrap/BUILD.bazel" <<'BUILD'
alias(
    name = "git-lfs",
    actual = "@git_lfs//:git-lfs",
)
BUILD

    (
        cd "$bootstrap"
        run_bazelisk --output_user_root="$BAZEL_LFS_ROOT"             build "${BAZEL_OFFLINE_FLAGS[@]}" //:git-lfs
    )
    local output_base
    output_base="$(
        cd "$bootstrap"
        run_bazelisk --output_user_root="$BAZEL_LFS_ROOT" info output_base
    )"
    local binary="$output_base/external/git_lfs/git-lfs"
    if [[ ! -x "$binary" ]]; then
        echo "Pinned Git LFS bootstrap did not produce an executable" >&2
        exit 1
    fi
    printf '%s\n' "$binary"
}

GIT_LFS_BIN="$(bootstrap_git_lfs)"
GIT_LFS_PATH="$(dirname "$GIT_LFS_BIN")"

if [[ ! -d "$SOURCE_DIR/.git" ]]; then
    if [[ "$OFFLINE" == "1" ]]; then
        echo "Offline source checkout is missing: $SOURCE_DIR" >&2
        exit 1
    fi
    mkdir -p "$(dirname "$SOURCE_DIR")"
    git clone --branch "$LITERTLM_GIT_TAG" --depth 1         "$LITERTLM_REPOSITORY" "$SOURCE_DIR"
fi

ACTUAL_COMMIT="$(git -C "$SOURCE_DIR" rev-parse HEAD)"
if [[ "$ACTUAL_COMMIT" != "$LITERTLM_GIT_COMMIT" ]]; then
    echo "LiteRT-LM checkout mismatch: expected $LITERTLM_GIT_COMMIT, got $ACTUAL_COMMIT" >&2
    exit 1
fi

if [[ "$OFFLINE" != "1" ]]; then
    PATH="$GIT_LFS_PATH:$PATH" git -C "$SOURCE_DIR" lfs install --local
    PATH="$GIT_LFS_PATH:$PATH" git -C "$SOURCE_DIR" lfs pull         --include="prebuilt/android_arm64/**"
fi

CONSTRAINT_LIBRARY="$SOURCE_DIR/prebuilt/android_arm64/libGemmaModelConstraintProvider.so"
if ! "$READELF" -h "$CONSTRAINT_LIBRARY" |
        grep -Eq 'Machine:[[:space:]]+AArch64'; then
    echo "LiteRT-LM Android LFS objects are missing or invalid" >&2
    exit 1
fi

if [[ "$SKIP_NATIVE" != "1" ]]; then
    (
        cd "$SOURCE_DIR"
        ANDROID_NDK_HOME="$ANDROID_NDK_ARG"             run_bazelisk --output_user_root="$BAZEL_OUTPUT_ROOT"             build "${BAZEL_OFFLINE_FLAGS[@]}"             --jobs="$JOBS" --config=android_arm64             //c:litert-lm             @litert//litert/vendors/google_tensor/dispatch:dispatch_api_so
    )
fi

RUNTIME_LIBRARY="$SOURCE_DIR/bazel-bin/c/liblitert-lm.so"
DISPATCH_LIBRARY="$SOURCE_DIR/bazel-bin/external/litert/litert/vendors/google_tensor/dispatch/libLiteRtDispatch_GoogleTensor.so"
for native_file in "$RUNTIME_LIBRARY" "$DISPATCH_LIBRARY" "$CONSTRAINT_LIBRARY"; do
    if ! "$READELF" -h "$native_file" |
            grep -Eq 'Machine:[[:space:]]+AArch64'; then
        echo "Missing Android ARM64 native artifact: $native_file" >&2
        exit 1
    fi
    if "$READELF" -d "$native_file" |
            grep -Eiq 'openblas|mkl|gfortran|quadmath|libc[.]so[.]6|ld-linux'; then
        echo "Forbidden BLAS/host dependency in $native_file" >&2
        exit 1
    fi
done

if [[ "$SKIP_JAVA" != "1" ]]; then
    "$MAVEN_CMD" "${MAVEN_FLAGS[@]}" \
        -f "$PRESET_MODULE/pom.xml" \
        -DskipTests install

    # Ship the reusable source-SDZ compile/cache API inside the provider AAR.
    "$MAVEN_CMD" "${MAVEN_FLAGS[@]}" \
        -f "$SDX_MODEL_MODULE/pom.xml" \
        -DskipTests install

    "$MAVEN_CMD" "${MAVEN_FLAGS[@]}" \
        -f "$MODULE/pom.xml" \
        -Pandroid-arm64-google-tensor-g5 clean package \
        -DskipTests \
        -Dandroid.ndk="$ANDROID_NDK_ARG" \
        -Dlitertlm.source.dir="$SOURCE_DIR" \
        -Dlitertlm.runtime.dir="$(dirname "$RUNTIME_LIBRARY")" \
        -Dlitertlm.runtime.library="$RUNTIME_LIBRARY" \
        -Dlitertlm.dispatch.library="$DISPATCH_LIBRARY" \
        -Dlitertlm.constraint.library="$CONSTRAINT_LIBRARY"
fi

MODULE_AAR="$MODULE/target/sdx-chat-runtime-1.0.0-SNAPSHOT-android-arm64-google-tensor-g5.aar"
if [[ ! -s "$MODULE_AAR" ]]; then
    echo "Google Tensor G5 AAR was not produced: $MODULE_AAR" >&2
    exit 1
fi

FINAL_AAR="$DIST_DIR/sdx-chat-runtime-android-arm64-google-tensor-g5.aar"
cp "$MODULE_AAR" "$FINAL_AAR"
"$VERIFY_SCRIPT" --aar "$FINAL_AAR" --android-ndk "$ANDROID_NDK_ARG"
sha256sum "$FINAL_AAR" > "$FINAL_AAR.sha256"

echo "Google Tensor G5 AAR: $FINAL_AAR"
echo "SHA-256 manifest: $FINAL_AAR.sha256"
echo "Capability: direct LiteRT-LM NPU / Tensor_G5 / AOT INT8"
echo "Fallback policy: no CPU, NNAPI, GPU, OpenBLAS, or host fallback"
