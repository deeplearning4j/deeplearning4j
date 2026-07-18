#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage:
  verify-android-accelerator-aar.sh --aar <file> --variant <name>
      --native-library <basename> --accelerator <enum> --android-ndk <path>
      [--gpu-target <enum>] [--device-ready] [--adapter-name <libname.so>]

Verifies Android ARM64 packaging, SDX/JavaCPP/tokenizer contents, binding
metadata, accelerator adapter symbols, ELF dependency closure, and the absence
of OpenBLAS/MKL/GFortran or Linux-host runtime dependencies.
USAGE
}

AAR=""
VARIANT=""
NATIVE_LIBRARY=""
ACCELERATOR=""
GPU_TARGET="AUTO"
ANDROID_NDK_ARG="${ANDROID_NDK:-${ANDROID_NDK_ROOT:-${ANDROID_NDK_HOME:-}}}"
DEVICE_READY=0
ADAPTER_NAME=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --aar)
            AAR="${2:?missing value for --aar}"
            shift 2
            ;;
        --variant)
            VARIANT="${2:?missing value for --variant}"
            shift 2
            ;;
        --native-library)
            NATIVE_LIBRARY="${2:?missing value for --native-library}"
            shift 2
            ;;
        --accelerator)
            ACCELERATOR="${2:?missing value for --accelerator}"
            shift 2
            ;;
        --gpu-target)
            GPU_TARGET="${2:?missing value for --gpu-target}"
            shift 2
            ;;
        --android-ndk)
            ANDROID_NDK_ARG="${2:?missing value for --android-ndk}"
            shift 2
            ;;
        --device-ready)
            DEVICE_READY=1
            shift
            ;;
        --adapter-name)
            ADAPTER_NAME="${2:?missing value for --adapter-name}"
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

for value_name in AAR VARIANT NATIVE_LIBRARY ACCELERATOR ANDROID_NDK_ARG; do
    if [[ -z "${!value_name:-}" ]]; then
        echo "Missing required argument: $value_name" >&2
        exit 2
    fi
done
if [[ ! -s "$AAR" ]]; then
    echo "AAR not found or empty: $AAR" >&2
    exit 1
fi
if [[ "$DEVICE_READY" == "1" && "$GPU_TARGET" != "VULKAN" && -z "$ADAPTER_NAME" ]]; then
    echo "--device-ready requires --adapter-name unless --gpu-target VULKAN is selected" >&2
    exit 2
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
MAIN_LIBRARY="$NATIVE_DIR/lib$NATIVE_LIBRARY.so"

REQUIRED_FILES=(
    "$TMP_DIR/AndroidManifest.xml"
    "$BINDING_JSON"
    "$CLASSES_JAR"
    "$MAIN_LIBRARY"
    "$NATIVE_DIR/libjnisdx.so"
    "$NATIVE_DIR/libjnitokenizers.so"
    "$NATIVE_DIR/libtokenizers_wrapper.so"
    "$NATIVE_DIR/libtokenizers_ffi.so"
)
for required_file in "${REQUIRED_FILES[@]}"; do
    if [[ ! -s "$required_file" ]]; then
        echo "Required AAR entry is missing or empty: $required_file" >&2
        exit 1
    fi
done

python3 - "$BINDING_JSON" "$VARIANT" "$NATIVE_LIBRARY" "$ACCELERATOR" \
           "$GPU_TARGET" "$DEVICE_READY" "$ADAPTER_NAME" <<'PY'
import json
import pathlib
import sys

path, variant, native_library, accelerator, gpu_target, device_ready, adapter_name = sys.argv[1:]
data = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))

assert data.get("formatVersion") == 1, data
assert data.get("runtimeAbi") == 1, data
assert data.get("platform", {}).get("id") == "android-arm64", data
assert data.get("platform", {}).get("os") == "android", data
assert data.get("platform", {}).get("arch") == "arm64", data
assert data.get("platform", {}).get("androidAbi") == "arm64-v8a", data
assert data.get("variant") == variant, data
assert data.get("runtimeLibrary") == f"lib{native_library}.so", data
assert data.get("defaultAccelerator") == accelerator, data
if accelerator != "NONE":
    assert accelerator in data.get("supportedAccelerators", []), data
assert data.get("defaultGpuTarget") == gpu_target, data
if gpu_target != "AUTO":
    assert gpu_target in data.get("supportedGpuTargets", []), data

deps = data.get("runtimeDependencies", [])
banned = (
    "openblas", "mkl", "gfortran", "quadmath", "libc.so.6", "ld-linux",
    "nd4jcpu", "neuralnetworks",
)
assert not any(any(token in dep.lower() for token in banned) for dep in deps), deps
if device_ready == "1" and adapter_name:
    assert adapter_name in deps, (adapter_name, deps)
PY

AAR_ENTRIES="$(unzip -Z1 "$AAR")"
if grep -Eiq 'openblas|mkl|gfortran|quadmath|libc[.]so[.]6|ld-linux|nd4jcpu|neuralnetworks' <<<"$AAR_ENTRIES"; then
    echo "AAR contains a forbidden host/BLAS runtime" >&2
    exit 1
fi

CLASS_ENTRIES="$(unzip -Z1 "$CLASSES_JAR")"
for class_name in \
    org/nd4j/dsp/model/SdxModelCache.class \
    org/nd4j/dsp/model/SdxTargetProfile.class \
    org/nd4j/dsp/runtime/SdxRuntime.class \
    org/nd4j/dsp/runtime/SdxTextSession.class \
    org/nd4j/dsp/runtime/bindings/SdxNative.class \
    org/eclipse/deeplearning4j/tokenizers/NativeTokenizer.class; do
    if ! grep -Fxq "$class_name" <<<"$CLASS_ENTRIES"; then
        echo "Required Java API class missing: $class_name" >&2
        exit 1
    fi
done
if grep -Eq '^org/nd4j/dsp/runtime/SdxLlm[^/]*[.]class$' <<<"$CLASS_ENTRIES"; then
    echo "Legacy JNA/Graal SdxLlm class leaked into the Android AAR" >&2
    exit 1
fi

if ! "$READELF" -h "$MAIN_LIBRARY" | grep -Eq 'Machine:[[:space:]]+AArch64'; then
    echo "Primary runtime is not an Android AArch64 ELF: $MAIN_LIBRARY" >&2
    exit 1
fi

DYNAMIC_SYMBOLS="$("$READELF" --dyn-syms --wide "$MAIN_LIBRARY")"
for symbol in \
    sdxGetRuntimeAbiVersion \
    sdxCreateRuntime \
    sdxDestroyRuntime \
    sdxLoadBundle \
    sdxUnloadModel \
    sdxCreateContextWithOptions \
    sdxDestroyContext \
    sdxRunAllocating \
    sdxGetExecutionReport \
    sdxCreateGenerationSession \
    sdxGenerationGenerate \
    sdxGenerationContinue \
    sdxDestroyGenerationSession; do
    if ! grep -Eq "[[:space:]]$symbol(@[^[:space:]]*)?$" <<<"$DYNAMIC_SYMBOLS"; then
        echo "Required SDX C ABI symbol missing from $MAIN_LIBRARY: $symbol" >&2
        exit 1
    fi
done

MAIN_NEEDED="$("$READELF" -d "$MAIN_LIBRARY")"
if [[ "$GPU_TARGET" == "VULKAN" ]]; then
    if ! grep -Eq 'Shared library: \[libvulkan[.]so\]' <<<"$MAIN_NEEDED"; then
        echo "Vulkan runtime is not linked to Android's libvulkan.so" >&2
        exit 1
    fi
    if grep -Eiq 'libneuralnetworks[.]so|libnd4jcpu[.]so' <<<"$MAIN_NEEDED"; then
        echo "Vulkan runtime links an alternate NNAPI/CPU backend" >&2
        exit 1
    fi
fi

is_android_system_library() {
    case "$1" in
        libc.so|libdl.so|libm.so|liblog.so|libz.so|libandroid.so|\
        libvulkan.so|libEGL.so|libGLESv2.so|libjnigraphics.so|\
        libmediandk.so|libnativewindow.so|libOpenSLES.so|libaaudio.so)
            return 0
            ;;
    esac
    return 1
}

shopt -s nullglob
NATIVE_LIBRARIES=("$NATIVE_DIR"/*.so)
shopt -u nullglob
if [[ ${#NATIVE_LIBRARIES[@]} -eq 0 ]]; then
    echo "No Android native libraries found in the AAR" >&2
    exit 1
fi

for native_file in "${NATIVE_LIBRARIES[@]}"; do
    if ! "$READELF" -h "$native_file" | grep -Eq 'Machine:[[:space:]]+AArch64'; then
        echo "Non-AArch64 native dependency in AAR: $native_file" >&2
        exit 1
    fi

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
            echo "Unpackaged native dependency $needed required by $(basename "$native_file")" >&2
            exit 1
        fi
    done < <("$READELF" -d "$native_file" |
              sed -n 's/.*Shared library: \[\([^]]*\)\].*/\1/p')
done

if [[ "$DEVICE_READY" == "1" && -n "$ADAPTER_NAME" ]]; then
    ADAPTER_LIBRARY="$NATIVE_DIR/$ADAPTER_NAME"
    if [[ ! -s "$ADAPTER_LIBRARY" ]]; then
        echo "Device-ready adapter is missing: $ADAPTER_NAME" >&2
        exit 1
    fi

    ADAPTER_SYMBOLS="$("$READELF" --dyn-syms --wide "$ADAPTER_LIBRARY")"
    for symbol in \
        hexmlir_get_device_count \
        hexmlir_init_device \
        hexmlir_release_device \
        hexmlir_allocate_tcm \
        hexmlir_free_tcm \
        hexmlir_get_tcm_capacity \
        hexmlir_allocate_ddr \
        hexmlir_free_ddr \
        hexmlir_release_kernel \
        hexmlir_dispatch_kernel \
        hexmlir_wait_for_completion \
        hexmlir_dma_host_to_tcm \
        hexmlir_dma_tcm_to_host \
        hexmlir_dma_copy; do
        if ! grep -Eq "[[:space:]]$symbol(@[^[:space:]]*)?$" <<<"$ADAPTER_SYMBOLS"; then
            echo "Vendor adapter symbol missing from $ADAPTER_NAME: $symbol" >&2
            exit 1
        fi
    done
    if ! grep -Eq '[[:space:]]hexmlir_(load|compile)_kernel(@[^[:space:]]*)?$' \
            <<<"$ADAPTER_SYMBOLS"; then
        echo "Vendor adapter exports neither hexmlir_load_kernel nor hexmlir_compile_kernel" >&2
        exit 1
    fi
fi

sha256sum "$AAR"
echo "Verified Android ARM64 accelerator AAR: $AAR"
echo "Variant: $VARIANT"
echo "Accelerator: $ACCELERATOR"
echo "GPU target: $GPU_TARGET"
echo "BLAS/host fallback dependencies: absent"
