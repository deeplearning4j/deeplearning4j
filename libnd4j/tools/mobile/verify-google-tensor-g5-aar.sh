#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage:
  verify-google-tensor-g5-aar.sh --aar <file> --android-ndk <path>

Verifies the direct Google Tensor G5 LiteRT-LM Android AAR. The check is
fail-closed: only the NPU runtime, Google dispatch library, JavaCPP bridge, and
required constraint provider may be packaged. CPU/NNAPI fallback, GPU runtime
extras, OpenBLAS, host libraries, and unresolved native dependencies are
rejected.
USAGE
}

AAR=""
ANDROID_NDK_ARG="${ANDROID_NDK:-${ANDROID_NDK_ROOT:-${ANDROID_NDK_HOME:-}}}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --aar)
            AAR="${2:?missing value for --aar}"
            shift 2
            ;;
        --android-ndk)
            ANDROID_NDK_ARG="${2:?missing value for --android-ndk}"
            shift 2
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

if [[ -z "$AAR" || -z "$ANDROID_NDK_ARG" ]]; then
    usage >&2
    exit 2
fi
if [[ ! -s "$AAR" ]]; then
    echo "AAR not found or empty: $AAR" >&2
    exit 1
fi

for command_name in unzip python3 sha256sum; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        echo "Required verification command not found: $command_name" >&2
        exit 1
    fi
done

case "$(uname -s)" in
    Linux) NDK_HOST_TAG=linux-x86_64 ;;
    Darwin) NDK_HOST_TAG=darwin-x86_64 ;;
    *)
        echo "Unsupported verification host: $(uname -s)" >&2
        exit 1
        ;;
esac
READELF="$ANDROID_NDK_ARG/toolchains/llvm/prebuilt/$NDK_HOST_TAG/bin/llvm-readelf"
if [[ ! -x "$READELF" ]]; then
    echo "NDK llvm-readelf not found: $READELF" >&2
    exit 1
fi

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT
unzip -q "$AAR" -d "$TMP_DIR"

NATIVE_DIR="$TMP_DIR/jni/arm64-v8a"
BINDING_JSON="$TMP_DIR/binding.json"
CLASSES_JAR="$TMP_DIR/classes.jar"
MAIN_LIBRARY="$NATIVE_DIR/liblitert-lm.so"
DISPATCH_LIBRARY="$NATIVE_DIR/libLiteRtDispatch_GoogleTensor.so"
CONSTRAINT_LIBRARY="$NATIVE_DIR/libGemmaModelConstraintProvider.so"
JAVACPP_LIBRARY="$NATIVE_DIR/libjnilitertlm.so"

REQUIRED_FILES=(
    "$TMP_DIR/AndroidManifest.xml"
    "$BINDING_JSON"
    "$CLASSES_JAR"
    "$MAIN_LIBRARY"
    "$DISPATCH_LIBRARY"
    "$CONSTRAINT_LIBRARY"
    "$JAVACPP_LIBRARY"
)
for required_file in "${REQUIRED_FILES[@]}"; do
    if [[ ! -s "$required_file" ]]; then
        echo "Required Tensor G5 AAR entry is missing: $required_file" >&2
        exit 1
    fi
done

python3 - "$BINDING_JSON" <<'PY'
import json
import pathlib
import sys

data = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
assert data.get("formatVersion") == 1, data
assert data.get("provider") == "litert-lm", data
assert data.get("providerVersion") == "0.14.0", data
assert data.get("variant") == "google-tensor-g5", data
assert data.get("platform", {}).get("id") == "android-arm64", data
assert data.get("platform", {}).get("androidAbi") == "arm64-v8a", data
assert data.get("hardware", {}).get("soc") == "Tensor_G5", data
assert data.get("hardware", {}).get("accelerator") == "Tensor TPU", data
assert data.get("backend") == "npu", data
assert data.get("modelFormat") == "litertlm", data
assert data.get("quantization") == "int8-per-channel", data
assert data.get("runtimeLibrary") == "liblitert-lm.so", data
assert data.get("javaCppLibrary") == "libjnilitertlm.so", data
policy = data.get("executionPolicy", {})
assert policy.get("aotOnly") is True, policy
assert policy.get("deviceOnly") is True, policy
assert policy.get("hostFallback") is False, policy
assert policy.get("blas") is False, policy
assert policy.get("jit") is False, policy
PY

AAR_ENTRIES="$(unzip -Z1 "$AAR")"
if grep -Eiq 'openblas|mkl|gfortran|quadmath|libc[.]so[.]6|ld-linux|libpthread[.]so[.]0'         <<<"$AAR_ENTRIES"; then
    echo "AAR contains a forbidden BLAS or host runtime" >&2
    exit 1
fi
if grep -Eiq 'opencl|webgpu|vulkan|nnapi|xnnpack' <<<"$AAR_ENTRIES"; then
    echo "AAR contains a non-TPU accelerator runtime" >&2
    exit 1
fi

CLASS_ENTRIES="$(unzip -Z1 "$CLASSES_JAR")"
for class_name in     org/nd4j/dsp/runtime/litertlm/SdxLiteRtLmChatSession.class     org/nd4j/dsp/runtime/litertlm/SdxLiteRtLmChatSession\$Builder.class     org/nd4j/dsp/runtime/litertlm/bindings/LiteRtLmNative.class; do
    if ! grep -Fxq "$class_name" <<<"$CLASS_ENTRIES"; then
        echo "Required JavaCPP API class missing: $class_name" >&2
        exit 1
    fi
done

shopt -s nullglob
NATIVE_LIBRARIES=("$NATIVE_DIR"/*.so)
shopt -u nullglob
if [[ ${#NATIVE_LIBRARIES[@]} -ne 4 ]]; then
    echo "Expected exactly four Tensor G5 native libraries, found ${#NATIVE_LIBRARIES[@]}" >&2
    exit 1
fi
for native_file in "${NATIVE_LIBRARIES[@]}"; do
    case "$(basename "$native_file")" in
        liblitert-lm.so|libLiteRtDispatch_GoogleTensor.so|        libGemmaModelConstraintProvider.so|libjnilitertlm.so)
            ;;
        *)
            echo "Unexpected native library in direct TPU AAR: $native_file" >&2
            exit 1
            ;;
    esac
    if ! "$READELF" -h "$native_file" |
            grep -Eq 'Machine:[[:space:]]+AArch64'; then
        echo "Non-AArch64 library in Tensor G5 AAR: $native_file" >&2
        exit 1
    fi
done

MAIN_SYMBOLS="$("$READELF" --dyn-syms --wide "$MAIN_LIBRARY")"
for symbol in     litert_lm_engine_settings_create     litert_lm_engine_settings_set_litert_dispatch_lib_dir     litert_lm_engine_create     litert_lm_conversation_create     litert_lm_conversation_send_message     litert_lm_conversation_send_message_stream     litert_lm_conversation_cancel_process     litert_lm_conversation_get_token_count     litert_lm_engine_tokenize     litert_lm_engine_detokenize; do
    if ! grep -Eq "[[:space:]]$symbol(@[^[:space:]]*)?$" <<<"$MAIN_SYMBOLS"; then
        echo "LiteRT-LM C ABI symbol missing: $symbol" >&2
        exit 1
    fi
done

DISPATCH_SYMBOLS="$("$READELF" --dyn-syms --wide "$DISPATCH_LIBRARY")"
if ! grep -Eq '[[:space:]]LiteRtDispatchGetApi(@[^[:space:]]*)?$'         <<<"$DISPATCH_SYMBOLS"; then
    echo "Google Tensor dispatch ABI entry point is missing" >&2
    exit 1
fi

is_android_system_library() {
    case "$1" in
        libc.so|libdl.so|libm.so|liblog.so|libandroid.so|libz.so|        libEGL.so|libGLESv2.so|libGLESv3.so|libnativewindow.so|        libjnigraphics.so|libmediandk.so|libOpenSLES.so|libaaudio.so)
            return 0
            ;;
    esac
    return 1
}

for native_file in "${NATIVE_LIBRARIES[@]}"; do
    while IFS= read -r needed; do
        [[ -z "$needed" ]] && continue
        needed_lower="${needed,,}"
        if [[ "$needed_lower" =~ openblas|mkl|gfortran|quadmath|libc[.]so[.]6|ld-linux|libpthread[.]so[.]0 ]]; then
            echo "Forbidden dependency $needed required by $(basename "$native_file")" >&2
            exit 1
        fi
        if is_android_system_library "$needed"; then
            continue
        fi
        if [[ ! -s "$NATIVE_DIR/$needed" ]]; then
            echo "Unpackaged dependency $needed required by $(basename "$native_file")" >&2
            exit 1
        fi
    done < <("$READELF" -d "$native_file" |
              sed -n 's/.*Shared library: \[\([^]]*\)\].*/\1/p')
done

if ! "$READELF" -d "$MAIN_LIBRARY" |
        grep -Fq 'Shared library: [libGemmaModelConstraintProvider.so]'; then
    echo "LiteRT-LM runtime is not linked to the packaged constraint provider" >&2
    exit 1
fi

sha256sum "$AAR"
echo "Verified direct Google Tensor G5 Android AAR: $AAR"
echo "Provider: LiteRT-LM 0.14.0 / backend=npu / SoC=Tensor_G5"
echo "Execution: AOT-only, device-only, INT8 per-channel"
echo "BLAS, host fallback, and alternate accelerator runtimes: absent"
