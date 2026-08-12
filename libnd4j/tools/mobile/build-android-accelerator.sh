#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage:
  build-android-accelerator.sh --profile <profile.env> --android-ndk <path> [options]

Options:
  --output-root <path>  Build and distribution root
  --jobs <n>            Native/Cargo parallelism (default: 4)
  --offline             Forbid Cargo and Maven network access and disconnect CMake FetchContent
  --device-ready        Require and validate the profile's vendor adapter library
  --skip-tokenizers     Reuse already-built Rust/JavaCPP tokenizer artifacts
  --skip-native         Reuse an already-built native SDX Android AAR
  --skip-java           Reuse an already-built full JavaCPP SDX Android AAR
  -h, --help            Show this help

The profile selects an accelerator-only variant. CPU and BLAS-backed profiles are
rejected. Set MVN_CMD to choose a Maven executable. Qualcomm device-ready builds
also require HEXAGON_ADAPTER_LIBRARY to point at libhexagon_mlir_runtime.so.
Vulkan uses Android's system loader and requires no bundled vendor adapter.
Set SDX_NATIVE_OOM_MEMORY_THRESHOLD to override the serialized native-build
host-memory threshold (default: 90).
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIBND4J_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_ROOT="$(cd "$LIBND4J_DIR/.." && pwd)"
TOKENIZER_ROOT="$REPO_ROOT/nd4j/nd4j-tokenizers"
TOKENIZER_BUILD="$TOKENIZER_ROOT/libtokenizers/build-mobile-tokenizers.sh"
TOKENIZER_PRESET_MODULE="$TOKENIZER_ROOT/tokenizers-native-preset"
TOKENIZER_MODULE="$TOKENIZER_ROOT/tokenizers-native"
SDX_MODULE="$REPO_ROOT/nd4j/nd4j-backends/nd4j-backend-impls/nd4j-sdx"
SDX_MODEL_MODULE="$REPO_ROOT/nd4j/nd4j-backends/nd4j-backend-impls/nd4j-sdx-model"
SDX_PRESET_MODULE="$REPO_ROOT/nd4j/nd4j-backends/nd4j-backend-impls/nd4j-sdx-preset"
VERIFY_SCRIPT="$SCRIPT_DIR/verify-android-accelerator-aar.sh"

PROFILE=""
ANDROID_NDK_ARG="${ANDROID_NDK:-${ANDROID_NDK_ROOT:-${ANDROID_NDK_HOME:-}}}"
OUTPUT_ROOT=""
JOBS="${JOBS:-4}"
OFFLINE=0
DEVICE_READY=0
SKIP_TOKENIZERS=0
SKIP_NATIVE=0
SKIP_JAVA=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --profile)
            PROFILE="${2:?missing value for --profile}"
            shift 2
            ;;
        --android-ndk)
            ANDROID_NDK_ARG="${2:?missing value for --android-ndk}"
            shift 2
            ;;
        --output-root)
            OUTPUT_ROOT="${2:?missing value for --output-root}"
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
        --device-ready)
            DEVICE_READY=1
            shift
            ;;
        --skip-tokenizers)
            SKIP_TOKENIZERS=1
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

if [[ -z "$PROFILE" ]]; then
    echo "--profile is required" >&2
    exit 2
fi
if [[ ! -f "$PROFILE" ]]; then
    echo "Profile not found: $PROFILE" >&2
    exit 1
fi
if [[ -z "$ANDROID_NDK_ARG" || ! -f "$ANDROID_NDK_ARG/build/cmake/android.toolchain.cmake" ]]; then
    echo "--android-ndk must point at an installed Android NDK" >&2
    exit 1
fi
if [[ ! "$JOBS" =~ ^[1-9][0-9]*$ ]]; then
    echo "--jobs must be a positive integer" >&2
    exit 2
fi

# Profiles are declarative inputs checked into tools/mobile/profiles.
# shellcheck disable=SC1090
source "$PROFILE"

SDX_GPU_TARGET="${SDX_GPU_TARGET:-AUTO}"

for required in SDX_VARIANT SDX_CHIP SDX_NATIVE_LIBRARY SDX_ACCELERATOR \
                SDX_ANDROID_API SDX_ANDROID_ABI SDX_DEVICE_ONLY \
                SDX_AOT_ONLY SDX_ALLOW_HOST_FALLBACK SDX_EXPECT_BLAS; do
    if [[ -z "${!required:-}" ]]; then
        echo "Profile is missing $required: $PROFILE" >&2
        exit 1
    fi
done

if [[ "$SDX_DEVICE_ONLY" != "1" || "$SDX_EXPECT_BLAS" != "0" ||
      "$SDX_ALLOW_HOST_FALLBACK" != "0" ]]; then
    echo "Accelerator profiles must be device-only, BLAS-free, and fail closed" >&2
    exit 1
fi
case "$SDX_CHIP" in
    vulkan|hexagon|google-tpu|nnapi) ;;
    *)
        echo "Unsupported Android device accelerator chip: $SDX_CHIP" >&2
        exit 1
        ;;
esac
if [[ "$SDX_ANDROID_ABI" != "arm64-v8a" ]]; then
    echo "Only Android arm64-v8a is supported by this mobile pipeline" >&2
    exit 1
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-$LIBND4J_DIR/build/mobile/$SDX_VARIANT}"
NATIVE_BUILD_DIR="$OUTPUT_ROOT/native"
DIST_DIR="$OUTPUT_ROOT/dist"
mkdir -p "$NATIVE_BUILD_DIR" "$DIST_DIR"

MVN_CMD="${MVN_CMD:-mvn}"
if ! command -v "$MVN_CMD" >/dev/null 2>&1; then
    echo "Maven executable not found: $MVN_CMD" >&2
    exit 1
fi
if [[ -z "${JAVA_HOME:-}" || ! -x "$JAVA_HOME/bin/java" ]]; then
    echo "JAVA_HOME must point at the JDK used for the Maven packaging build" >&2
    exit 1
fi
MVN_REAL="$(realpath -e -- "$(command -v "$MVN_CMD")")"
JAVA_HOME_REAL="$(realpath -e -- "$JAVA_HOME")"
MAVEN_SHA256="$(sha256sum "$MVN_REAL" | cut -d ' ' -f 1)"
MAVEN_VERSION_SHA256="$(
    { env JAVA_HOME="$JAVA_HOME_REAL" PATH="$JAVA_HOME_REAL/bin:$PATH" "$MVN_REAL" --version; } 2>&1 |
        sha256sum | cut -d ' ' -f 1
)"
JAVA_VERSION_SHA256="$(
    { "$JAVA_HOME_REAL/bin/java" -version; } 2>&1 | sha256sum | cut -d ' ' -f 1
)"
for command_name in cmake cargo rustup unzip zip sha256sum realpath mktemp git stat sort find grep; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        echo "Required build command not found: $command_name" >&2
        exit 1
    fi
done

sha256_file() {
    sha256sum "$1" | cut -d ' ' -f 1
}

tree_manifest_sha256() {
    local root="$1"
    local file relative mode digest
    [[ -d "$root" ]] || {
        echo "Tree manifest root is missing: $root" >&2
        return 1
    }
    if find "$root" -type l -print -quit | grep -q .; then
        echo "Tree manifest root contains a symlink: $root" >&2
        return 1
    fi
    {
        while IFS= read -r -d '' file; do
            relative="${file#"$root"/}"
            mode="$(stat -c '%a' "$file")"
            digest="$(sha256_file "$file")"
            printf '%s\0%s\0%s\0' "$relative" "$mode" "$digest"
        done < <(find "$root" -type f -print0 | LC_ALL=C sort -z)
    } | sha256sum | cut -d ' ' -f 1
}

module_source_manifest_sha256() {
    local module="$1"
    local relative_root relative file mode digest
    relative_root="${module#"$REPO_ROOT"/}"
    [[ "$relative_root" != "$module" ]] || {
        echo "Module is outside the source tree: $module" >&2
        return 1
    }
    {
        git -C "$REPO_ROOT" ls-files -z --cached --others --exclude-standard -- "$relative_root" |
            LC_ALL=C sort -z |
            while IFS= read -r -d '' relative; do
                file="$REPO_ROOT/$relative"
                [[ -f "$file" ]] || continue
                mode="$(stat -c '%a' "$file")"
                digest="$(sha256_file "$file")"
                printf '%s\0%s\0%s\0' "$relative" "$mode" "$digest"
            done
    } | sha256sum | cut -d ' ' -f 1
}

prepare_fresh_maven_target() {
    local id="$1"
    local module="$2"
    local target="$module/target"
    [[ ! -L "$target" ]] || {
        echo "Refusing symlinked Maven target for $id: $target" >&2
        return 1
    }
    if [[ -e "$target" ]]; then
        mv -- "$target" "$MAVEN_TARGET_QUARANTINE/$id"
    fi
    mkdir -p "$target"
    echo "Fresh Maven target for $id; previous output quarantined under $MAVEN_TARGET_QUARANTINE"
}

record_fresh_maven_build() {
    local id="$1"
    local module="$2"
    local classes="$module/target/classes"
    local classes_real source_sha classes_sha
    [[ -d "$classes" ]] || {
        echo "Fresh Maven build produced no classes for $id: $classes" >&2
        return 1
    }
    classes_real="$(realpath -e -- "$classes")"
    [[ "$id" != *[[:space:]]* && "$classes_real" != *[[:space:]]* ]] || {
        echo "Fresh Maven provenance fields cannot contain whitespace: $id $classes_real" >&2
        return 1
    }
    source_sha="$(module_source_manifest_sha256 "$module")"
    classes_sha="$(tree_manifest_sha256 "$classes_real")"
    printf '%s %s %s %s\n' "$id" "$source_sha" "$classes_real" "$classes_sha" >>"$FRESH_JAVA_BUILDS_TMP"
}

archive_member_sha256() {
    local archive="$1"
    local member="$2"
    if ! unzip -Z1 "$archive" | grep -Fxq "$member"; then
        echo "Required archive member is missing from $archive: $member" >&2
        return 1
    fi
    unzip -p "$archive" "$member" | sha256sum | cut -d ' ' -f 1
}

source_manifest_sha256() {
    local relative file mode digest
    local -a roots=(
        libnd4j
        nd4j/sdx-aot
        nd4j/nd4j-tokenizers
        nd4j/nd4j-backends/nd4j-backend-impls/nd4j-sdx
        nd4j/nd4j-backends/nd4j-backend-impls/nd4j-sdx-model
        nd4j/nd4j-backends/nd4j-backend-impls/nd4j-sdx-preset
    )
    local -a excludes=(
        ':(top,exclude)libnd4j/cmake/tests/**'
    )
    {
        # CMake contract tests are not inputs to the Android provider build. Keep
        # unrelated test edits out of the production receipt while retaining every
        # native, Java, profile, and build-script input used by this build.
        git -C "$REPO_ROOT" ls-files -z --cached --others --exclude-standard -- "${roots[@]}" "${excludes[@]}" |
            LC_ALL=C sort -z |
            while IFS= read -r -d '' relative; do
                file="$REPO_ROOT/$relative"
                [[ -f "$file" ]] || continue
                mode="$(stat -c '%a' "$file")"
                digest="$(sha256_file "$file")"
                printf '%s\0%s\0%s\0' "$relative" "$mode" "$digest"
            done
    } | sha256sum | cut -d ' ' -f 1
}

declare -A RECEIPT_VALUES=()
load_strict_receipt() {
    local receipt="$1"
    local line key value
    RECEIPT_VALUES=()
    [[ -s "$receipt" ]] || {
        echo "Build receipt is missing or empty: $receipt" >&2
        return 1
    }
    while IFS= read -r line || [[ -n "$line" ]]; do
        [[ "$line" == *=* ]] || {
            echo "Malformed build receipt line in $receipt: $line" >&2
            return 1
        }
        key="${line%%=*}"
        value="${line#*=}"
        [[ "$key" =~ ^[a-z][a-z0-9_]*$ && -n "$value" ]] || {
            echo "Invalid build receipt field in $receipt: $line" >&2
            return 1
        }
        case "$key" in
            format|stage|variant|artifact|sha256|inputs_sha256|source_manifest_sha256|profile_sha256|build_script_sha256|ndk_revision_sha256|android_api|android_abi|chip|helpers|required_accelerator_device|provider_member|provider_sha256|arm_compute_member|arm_compute_sha256|native_artifact|native_sha256|native_receipt_sha256|full_source_artifact|full_source_sha256|classes_sha256|jni_bridge_member|jni_bridge_sha256) ;;
            *)
                echo "Unknown build receipt field in $receipt: $key" >&2
                return 1
                ;;
        esac
        [[ ! -v 'RECEIPT_VALUES[$key]' ]] || {
            echo "Duplicate build receipt field in $receipt: $key" >&2
            return 1
        }
        RECEIPT_VALUES["$key"]="$value"
    done <"$receipt"
}

require_receipt_fields() {
    local receipt="$1"
    shift
    local key
    for key in "$@"; do
        [[ -v 'RECEIPT_VALUES[$key]' ]] || {
            echo "Build receipt $receipt omitted required field: $key" >&2
            return 1
        }
    done
}

MAVEN_OFFLINE=()
TOKENIZER_OFFLINE=()
if [[ "$OFFLINE" == "1" ]]; then
    MAVEN_OFFLINE=(-o)
    TOKENIZER_OFFLINE=(--offline)
    export SDX_OFFLINE=ON
fi

ADAPTER_LIBRARY=""
if [[ "$SDX_VARIANT" == "hexagon" ]]; then
    ADAPTER_LIBRARY="${HEXAGON_ADAPTER_LIBRARY:-}"
    if [[ -n "$ADAPTER_LIBRARY" && ! -f "$ADAPTER_LIBRARY" ]]; then
        echo "HEXAGON_ADAPTER_LIBRARY does not exist: $ADAPTER_LIBRARY" >&2
        exit 1
    fi
    if [[ "$DEVICE_READY" == "1" && -z "$ADAPTER_LIBRARY" ]]; then
        echo "--device-ready requires HEXAGON_ADAPTER_LIBRARY" >&2
        exit 1
    fi
fi
if [[ -n "$ADAPTER_LIBRARY" ]]; then
    export SDX_EXTRA_RUNTIME_DEPENDENCY_FILES="$ADAPTER_LIBRARY"
    DEVICE_READY=1
else
    unset SDX_EXTRA_RUNTIME_DEPENDENCY_FILES || true
fi
if [[ "$SDX_VARIANT" == "vulkan" ]]; then
    DEVICE_READY=1
fi
if [[ "$SDX_VARIANT" == "tensor-g3" ]]; then
    export SD_NNAPI_REQUIRED_DEVICE_NAME="${SDX_REQUIRED_ACCELERATOR_DEVICE:-google-edgetpu}"
    # Tensor G3 uses the prevalidated Android/NDK ACL DSO. The generic
    # armcompute_install artifact is the Linux/libstdc++ package and cannot
    # satisfy an Android libc++ consumer. Keep the pipe escaped because
    # buildnativeoperations.sh evaluates CMAKE_ARGUMENTS as shell words.
    export CMAKE_ARGUMENTS="${CMAKE_ARGUMENTS:-} -DHELPERS_armcompute=ON -DSDX_NNAPI_TENSOR_G3_HYBRID=ON -DSDX_EXTRA_RUNTIME_DEPENDENCY_FILES:STRING=$NATIVE_BUILD_DIR/tensor_g3_armcompute_install/lib/armv8a-neon/libarm_compute.so"
fi

if [[ "$SKIP_TOKENIZERS" == "1" || "$SKIP_JAVA" == "1" ]]; then
    echo "--skip-tokenizers and --skip-java are disabled for provenance format 2 builds." >&2
    echo "A full Java/tokenizer rebuild is required so no unreceipted target output can be selected." >&2
    exit 1
fi

PROFILE_REAL="$(realpath -e -- "$PROFILE")"
PROFILE_SHA256="$(sha256_file "$PROFILE_REAL")"
BUILD_SCRIPT_SHA256="$(sha256_file "$0")"
NDK_REVISION_FILE="$ANDROID_NDK_ARG/source.properties"
[[ -s "$NDK_REVISION_FILE" ]] || {
    echo "Android NDK revision file is missing: $NDK_REVISION_FILE" >&2
    exit 1
}
NDK_REVISION_SHA256="$(sha256_file "$NDK_REVISION_FILE")"
SOURCE_MANIFEST_SHA256="$(source_manifest_sha256)"
MAVEN_TARGET_QUARANTINE="$(mktemp -d "$OUTPUT_ROOT/quarantined-maven-targets.XXXXXX")"
FRESH_JAVA_BUILDS_TMP="$(mktemp "$DIST_DIR/fresh-java-builds.tmp.XXXXXX")"
NATIVE_HELPERS="none"
REQUIRED_ACCELERATOR_DEVICE="none"
if [[ "$SDX_VARIANT" == "tensor-g3" ]]; then
    NATIVE_HELPERS="nnapi,armcompute"
    REQUIRED_ACCELERATOR_DEVICE="$SD_NNAPI_REQUIRED_DEVICE_NAME"
fi
NATIVE_INPUTS_SHA256="$(
    printf '%s\n' \
        "source_manifest_sha256=$SOURCE_MANIFEST_SHA256" \
        "profile_sha256=$PROFILE_SHA256" \
        "build_script_sha256=$BUILD_SCRIPT_SHA256" \
        "ndk_revision_sha256=$NDK_REVISION_SHA256" \
        "variant=$SDX_VARIANT" \
        "android_api=$SDX_ANDROID_API" \
        "android_abi=$SDX_ANDROID_ABI" \
        "chip=$SDX_CHIP" \
        "helpers=$NATIVE_HELPERS" \
        "required_accelerator_device=$REQUIRED_ACCELERATOR_DEVICE" |
        sha256sum | cut -d ' ' -f 1
)"

if [[ "$SKIP_TOKENIZERS" != "1" ]]; then
    "$TOKENIZER_BUILD" \
        --platform android-arm64 \
        --android-ndk "$ANDROID_NDK_ARG" \
        --android-api "$SDX_ANDROID_API" \
        --jobs "$JOBS" \
        "${TOKENIZER_OFFLINE[@]}"

    # Maven's clean phase is deliberately forbidden in this pipeline. Move each
    # old target aside, then build into a newly-created target so removed classes
    # cannot survive while interrupted output remains recoverable.
    prepare_fresh_maven_target "tokenizers-native-preset" "$TOKENIZER_PRESET_MODULE"
    "$MVN_REAL" "${MAVEN_OFFLINE[@]}" \
        -f "$TOKENIZER_PRESET_MODULE/pom.xml" \
        -DskipTests install
    record_fresh_maven_build "tokenizers-native-preset" "$TOKENIZER_PRESET_MODULE"

    prepare_fresh_maven_target "tokenizers-native" "$TOKENIZER_MODULE"
    "$MVN_REAL" "${MAVEN_OFFLINE[@]}" \
        -f "$TOKENIZER_MODULE/pom.xml" \
        -Pandroid-arm64 install \
        -DskipTests \
        -Dandroid.ndk="$ANDROID_NDK_ARG" \
        -Dandroid.api="$SDX_ANDROID_API"
    record_fresh_maven_build "tokenizers-native" "$TOKENIZER_MODULE"
fi

NATIVE_AAR="$NATIVE_BUILD_DIR/sdx-runtime-sdk/dist/sdx-runtime-android-arm64-$SDX_VARIANT.aar"
NATIVE_RECEIPT="$NATIVE_AAR.build-receipt"

if [[ "$SKIP_NATIVE" != "1" ]]; then
    # A failed or interrupted native rebuild must not leave an older receipt
    # authorizing whatever bytes happen to remain at the stable output path.
    rm -f -- "$NATIVE_RECEIPT"
    NATIVE_ARGS=(
        --platform android-arm64
        --android-abi "$SDX_ANDROID_ABI"
        --android-api "$SDX_ANDROID_API"
        --chip "$SDX_CHIP"
        --build-type release
        --output-path "$NATIVE_BUILD_DIR"
        --preprocess OFF
        # Mobile provider builds are deliberately serialized. The generic 80%
        # host-wide threshold can otherwise terminate a cache-hit-only build
        # because of unrelated workloads while tens of GiB remain available.
        --oom-memory-threshold "${SDX_NATIVE_OOM_MEMORY_THRESHOLD:-90}"
        -j "$JOBS"
    )
    if [[ "$SDX_VARIANT" == "tensor-g3" ]]; then
        NATIVE_ARGS+=(--helpers nnapi,armcompute)
    fi
    if [[ -n "${SDX_DATATYPES:-}" ]]; then
        NATIVE_ARGS+=(--datatypes "$SDX_DATATYPES")
    fi
    if [[ -n "${SDX_OPERATIONS:-}" ]]; then
        NATIVE_ARGS+=(--operations "$SDX_OPERATIONS")
    fi

    ANDROID_NDK="$ANDROID_NDK_ARG" \
    ANDROID_API="$SDX_ANDROID_API" \
    BUILD_WITH_JAVA=OFF \
    PREPROCESS=OFF \
        "$LIBND4J_DIR/buildnativeoperations.sh" "${NATIVE_ARGS[@]}"

    cmake --build "$NATIVE_BUILD_DIR" \
        --target sdx_runtime_bindings \
        --parallel "$JOBS"
fi

if [[ ! -s "$NATIVE_AAR" ]]; then
    echo "Native SDX AAR was not produced: $NATIVE_AAR" >&2
    exit 1
fi

NATIVE_AAR_REAL="$(realpath -e -- "$NATIVE_AAR")"
NATIVE_AAR_SHA256="$(sha256_file "$NATIVE_AAR_REAL")"
NATIVE_PROVIDER_MEMBER="jni/arm64-v8a/lib$SDX_NATIVE_LIBRARY.so"
NATIVE_PROVIDER_SHA256="$(archive_member_sha256 "$NATIVE_AAR_REAL" "$NATIVE_PROVIDER_MEMBER")"
NATIVE_ARM_COMPUTE_MEMBER="none"
NATIVE_ARM_COMPUTE_SHA256="none"
if [[ "$SDX_VARIANT" == "tensor-g3" ]]; then
    NATIVE_ARM_COMPUTE_MEMBER="jni/arm64-v8a/libarm_compute.so"
    NATIVE_ARM_COMPUTE_SHA256="$(
        archive_member_sha256 "$NATIVE_AAR_REAL" "$NATIVE_ARM_COMPUTE_MEMBER"
    )"
fi
if [[ "$SKIP_NATIVE" != "1" ]]; then
    NATIVE_RECEIPT_TMP="$(mktemp "$NATIVE_RECEIPT.tmp.XXXXXX")"
    {
        printf 'format=2\n'
        printf 'stage=native\n'
        printf 'variant=%s\n' "$SDX_VARIANT"
        printf 'artifact=%s\n' "$NATIVE_AAR_REAL"
        printf 'sha256=%s\n' "$NATIVE_AAR_SHA256"
        printf 'inputs_sha256=%s\n' "$NATIVE_INPUTS_SHA256"
        printf 'source_manifest_sha256=%s\n' "$SOURCE_MANIFEST_SHA256"
        printf 'profile_sha256=%s\n' "$PROFILE_SHA256"
        printf 'build_script_sha256=%s\n' "$BUILD_SCRIPT_SHA256"
        printf 'ndk_revision_sha256=%s\n' "$NDK_REVISION_SHA256"
        printf 'android_api=%s\n' "$SDX_ANDROID_API"
        printf 'android_abi=%s\n' "$SDX_ANDROID_ABI"
        printf 'chip=%s\n' "$SDX_CHIP"
        printf 'helpers=%s\n' "$NATIVE_HELPERS"
        printf 'required_accelerator_device=%s\n' "$REQUIRED_ACCELERATOR_DEVICE"
        printf 'provider_member=%s\n' "$NATIVE_PROVIDER_MEMBER"
        printf 'provider_sha256=%s\n' "$NATIVE_PROVIDER_SHA256"
        printf 'arm_compute_member=%s\n' "$NATIVE_ARM_COMPUTE_MEMBER"
        printf 'arm_compute_sha256=%s\n' "$NATIVE_ARM_COMPUTE_SHA256"
    } >"$NATIVE_RECEIPT_TMP"
    mv -f -- "$NATIVE_RECEIPT_TMP" "$NATIVE_RECEIPT"
    echo "Native build receipt: $NATIVE_RECEIPT"
else
    if [[ ! -s "$NATIVE_RECEIPT" ]]; then
        echo "--skip-native requires a completed native build receipt: $NATIVE_RECEIPT" >&2
        echo "Rerun without --skip-native to rebuild and authorize the native AAR." >&2
        exit 1
    fi
    load_strict_receipt "$NATIVE_RECEIPT"
    require_receipt_fields "$NATIVE_RECEIPT" \
        format stage variant artifact sha256 inputs_sha256 source_manifest_sha256 \
        profile_sha256 build_script_sha256 ndk_revision_sha256 android_api android_abi \
        chip helpers required_accelerator_device provider_member provider_sha256 \
        arm_compute_member arm_compute_sha256
    if [[ "${RECEIPT_VALUES[format]}" != "2" ||
          "${RECEIPT_VALUES[stage]}" != "native" ||
          "${RECEIPT_VALUES[variant]}" != "$SDX_VARIANT" ||
          "${RECEIPT_VALUES[artifact]}" != "$NATIVE_AAR_REAL" ||
          "${RECEIPT_VALUES[sha256]}" != "$NATIVE_AAR_SHA256" ||
          "${RECEIPT_VALUES[inputs_sha256]}" != "$NATIVE_INPUTS_SHA256" ||
          "${RECEIPT_VALUES[source_manifest_sha256]}" != "$SOURCE_MANIFEST_SHA256" ||
          "${RECEIPT_VALUES[profile_sha256]}" != "$PROFILE_SHA256" ||
          "${RECEIPT_VALUES[build_script_sha256]}" != "$BUILD_SCRIPT_SHA256" ||
          "${RECEIPT_VALUES[ndk_revision_sha256]}" != "$NDK_REVISION_SHA256" ||
          "${RECEIPT_VALUES[android_api]}" != "$SDX_ANDROID_API" ||
          "${RECEIPT_VALUES[android_abi]}" != "$SDX_ANDROID_ABI" ||
          "${RECEIPT_VALUES[chip]}" != "$SDX_CHIP" ||
          "${RECEIPT_VALUES[helpers]}" != "$NATIVE_HELPERS" ||
          "${RECEIPT_VALUES[required_accelerator_device]}" != "$REQUIRED_ACCELERATOR_DEVICE" ||
          "${RECEIPT_VALUES[provider_member]}" != "$NATIVE_PROVIDER_MEMBER" ||
          "${RECEIPT_VALUES[provider_sha256]}" != "$NATIVE_PROVIDER_SHA256" ||
          "${RECEIPT_VALUES[arm_compute_member]}" != "$NATIVE_ARM_COMPUTE_MEMBER" ||
          "${RECEIPT_VALUES[arm_compute_sha256]}" != "$NATIVE_ARM_COMPUTE_SHA256" ]]; then
        echo "Native build receipt does not authorize the current AAR: $NATIVE_RECEIPT" >&2
        echo "Rerun without --skip-native to rebuild and refresh the receipt." >&2
        exit 1
    fi
    echo "Verified native build receipt: $NATIVE_RECEIPT"
fi

if [[ "$SKIP_JAVA" != "1" ]]; then
    # JavaCPP-generated bindings name their preset classes directly. Build the
    # preset from this checkout and package it into the AAR so R8 and Android do
    # not depend on whatever happens to be present in Maven local.
    prepare_fresh_maven_target "nd4j-sdx-preset" "$SDX_PRESET_MODULE"
    "$MVN_REAL" "${MAVEN_OFFLINE[@]}" \
        -f "$SDX_PRESET_MODULE/pom.xml" \
        -DskipTests install
    record_fresh_maven_build "nd4j-sdx-preset" "$SDX_PRESET_MODULE"

    # Mainline the source-SDZ compile/cache API into every provider AAR from a
    # quarantined, empty target directory.
    prepare_fresh_maven_target "nd4j-sdx-model" "$SDX_MODEL_MODULE"
    "$MVN_REAL" "${MAVEN_OFFLINE[@]}" \
        -f "$SDX_MODEL_MODULE/pom.xml" \
        -DskipTests install
    record_fresh_maven_build "nd4j-sdx-model" "$SDX_MODEL_MODULE"

    prepare_fresh_maven_target "nd4j-sdx" "$SDX_MODULE"
    "$MVN_REAL" "${MAVEN_OFFLINE[@]}" \
        -f "$SDX_MODULE/pom.xml" \
        -Pandroid-arm64 install \
        -DskipTests \
        -Dandroid.ndk="$ANDROID_NDK_ARG" \
        -Dandroid.api="$SDX_ANDROID_API" \
        -Dlibnd4j.outputPath="$NATIVE_BUILD_DIR" \
        -Dsdx.android.variant="$SDX_VARIANT" \
        -Dsdx.native.library="$SDX_NATIVE_LIBRARY"
    record_fresh_maven_build "nd4j-sdx" "$SDX_MODULE"
fi

shopt -s nullglob
AAR_CANDIDATES=("$SDX_MODULE/target"/sdx-runtime-*-android-arm64-"$SDX_VARIANT".aar)
shopt -u nullglob
if [[ ${#AAR_CANDIDATES[@]} -ne 1 || ! -s "${AAR_CANDIDATES[0]}" ]]; then
    echo "Expected one full JavaCPP AAR for $SDX_VARIANT, found ${#AAR_CANDIDATES[@]}" >&2
    exit 1
fi

FINAL_AAR="$DIST_DIR/sdx-runtime-android-arm64-$SDX_VARIANT.aar"
cp "${AAR_CANDIDATES[0]}" "$FINAL_AAR"

VERIFY_ARGS=(
    --aar "$FINAL_AAR"
    --variant "$SDX_VARIANT"
    --native-library "$SDX_NATIVE_LIBRARY"
    --accelerator "$SDX_ACCELERATOR"
    --gpu-target "$SDX_GPU_TARGET"
    --android-ndk "$ANDROID_NDK_ARG"
)
if [[ "$DEVICE_READY" == "1" ]]; then
    VERIFY_ARGS+=(--device-ready)
    if [[ -n "${SDX_ADAPTER_LIBRARY_NAME:-}" ]]; then
        VERIFY_ARGS+=(--adapter-name "$SDX_ADAPTER_LIBRARY_NAME")
    fi
fi
"$VERIFY_SCRIPT" "${VERIFY_ARGS[@]}"

FINAL_AAR_REAL="$(realpath -e -- "$FINAL_AAR")"
FINAL_AAR_SHA256="$(sha256_file "$FINAL_AAR_REAL")"
FULL_SOURCE_AAR_REAL="$(realpath -e -- "${AAR_CANDIDATES[0]}")"
FULL_SOURCE_AAR_SHA256="$(sha256_file "$FULL_SOURCE_AAR_REAL")"
FINAL_PROVIDER_SHA256="$(archive_member_sha256 "$FINAL_AAR_REAL" "$NATIVE_PROVIDER_MEMBER")"
[[ "$FINAL_PROVIDER_SHA256" == "$NATIVE_PROVIDER_SHA256" ]] || {
    echo "Full AAR does not contain the exact provider DSO from the receipted native AAR" >&2
    exit 1
}
FINAL_ARM_COMPUTE_SHA256="none"
if [[ "$NATIVE_ARM_COMPUTE_MEMBER" != "none" ]]; then
    FINAL_ARM_COMPUTE_SHA256="$(
        archive_member_sha256 "$FINAL_AAR_REAL" "$NATIVE_ARM_COMPUTE_MEMBER"
    )"
    [[ "$FINAL_ARM_COMPUTE_SHA256" == "$NATIVE_ARM_COMPUTE_SHA256" ]] || {
        echo "Full AAR does not contain the exact ARM Compute DSO from the receipted native AAR" >&2
        exit 1
    }
fi
FINAL_CLASSES_SHA256="$(archive_member_sha256 "$FINAL_AAR_REAL" "classes.jar")"
FINAL_JNI_BRIDGE_MEMBER="jni/arm64-v8a/libjnisdx.so"
FINAL_JNI_BRIDGE_SHA256="$(
    archive_member_sha256 "$FINAL_AAR_REAL" "$FINAL_JNI_BRIDGE_MEMBER"
)"
CURRENT_SOURCE_MANIFEST_SHA256="$(source_manifest_sha256)"
[[ "$CURRENT_SOURCE_MANIFEST_SHA256" == "$SOURCE_MANIFEST_SHA256" ]] || {
    echo "Source tree changed during Android accelerator build" >&2
    exit 1
}
FRESH_JAVA_BUILDS="$FINAL_AAR.fresh-java-builds"
mv -f -- "$FRESH_JAVA_BUILDS_TMP" "$FRESH_JAVA_BUILDS"
FRESH_JAVA_BUILDS_REAL="$(realpath -e -- "$FRESH_JAVA_BUILDS")"
FRESH_JAVA_BUILDS_SHA256="$(sha256_file "$FRESH_JAVA_BUILDS_REAL")"
NATIVE_RECEIPT_SHA256="$(sha256_file "$NATIVE_RECEIPT")"
FULL_INPUTS_SHA256="$(
    printf '%s\n' \
        "native_receipt_sha256=$NATIVE_RECEIPT_SHA256" \
        "native_sha256=$NATIVE_AAR_SHA256" \
        "source_manifest_sha256=$SOURCE_MANIFEST_SHA256" \
        "full_source_sha256=$FULL_SOURCE_AAR_SHA256" \
        "classes_sha256=$FINAL_CLASSES_SHA256" \
        "fresh_java_builds_sha256=$FRESH_JAVA_BUILDS_SHA256" \
        "maven_sha256=$MAVEN_SHA256" \
        "maven_version_sha256=$MAVEN_VERSION_SHA256" \
        "java_version_sha256=$JAVA_VERSION_SHA256" \
        "provider_sha256=$FINAL_PROVIDER_SHA256" \
        "arm_compute_sha256=$FINAL_ARM_COMPUTE_SHA256" \
        "jni_bridge_sha256=$FINAL_JNI_BRIDGE_SHA256" |
        sha256sum | cut -d ' ' -f 1
)"
FINAL_RECEIPT="$FINAL_AAR.build-receipt"
FINAL_RECEIPT_TMP="$(mktemp "$FINAL_RECEIPT.tmp.XXXXXX")"
{
    printf 'format=3\n'
    printf 'stage=full\n'
    printf 'variant=%s\n' "$SDX_VARIANT"
    printf 'artifact=%s\n' "$FINAL_AAR_REAL"
    printf 'sha256=%s\n' "$FINAL_AAR_SHA256"
    printf 'inputs_sha256=%s\n' "$FULL_INPUTS_SHA256"
    printf 'source_manifest_sha256=%s\n' "$SOURCE_MANIFEST_SHA256"
    printf 'profile_sha256=%s\n' "$PROFILE_SHA256"
    printf 'build_script_sha256=%s\n' "$BUILD_SCRIPT_SHA256"
    printf 'ndk_revision_sha256=%s\n' "$NDK_REVISION_SHA256"
    printf 'android_api=%s\n' "$SDX_ANDROID_API"
    printf 'android_abi=%s\n' "$SDX_ANDROID_ABI"
    printf 'chip=%s\n' "$SDX_CHIP"
    printf 'helpers=%s\n' "$NATIVE_HELPERS"
    printf 'required_accelerator_device=%s\n' "$REQUIRED_ACCELERATOR_DEVICE"
    printf 'native_artifact=%s\n' "$NATIVE_AAR_REAL"
    printf 'native_sha256=%s\n' "$NATIVE_AAR_SHA256"
    printf 'native_receipt_sha256=%s\n' "$NATIVE_RECEIPT_SHA256"
    printf 'full_source_artifact=%s\n' "$FULL_SOURCE_AAR_REAL"
    printf 'full_source_sha256=%s\n' "$FULL_SOURCE_AAR_SHA256"
    printf 'classes_sha256=%s\n' "$FINAL_CLASSES_SHA256"
    printf 'fresh_java_builds=%s\n' "$FRESH_JAVA_BUILDS_REAL"
    printf 'fresh_java_builds_sha256=%s\n' "$FRESH_JAVA_BUILDS_SHA256"
    printf 'maven=%s\n' "$MVN_REAL"
    printf 'maven_sha256=%s\n' "$MAVEN_SHA256"
    printf 'maven_version_sha256=%s\n' "$MAVEN_VERSION_SHA256"
    printf 'java_home=%s\n' "$JAVA_HOME_REAL"
    printf 'java_version_sha256=%s\n' "$JAVA_VERSION_SHA256"
    printf 'provider_member=%s\n' "$NATIVE_PROVIDER_MEMBER"
    printf 'provider_sha256=%s\n' "$FINAL_PROVIDER_SHA256"
    printf 'arm_compute_member=%s\n' "$NATIVE_ARM_COMPUTE_MEMBER"
    printf 'arm_compute_sha256=%s\n' "$FINAL_ARM_COMPUTE_SHA256"
    printf 'jni_bridge_member=%s\n' "$FINAL_JNI_BRIDGE_MEMBER"
    printf 'jni_bridge_sha256=%s\n' "$FINAL_JNI_BRIDGE_SHA256"
} >"$FINAL_RECEIPT_TMP"
mv -f -- "$FINAL_RECEIPT_TMP" "$FINAL_RECEIPT"

sha256sum "$FINAL_AAR" > "$FINAL_AAR.sha256"
echo "Android accelerator AAR: $FINAL_AAR"
echo "SHA-256 manifest: $FINAL_AAR.sha256"
echo "Full build receipt: $FINAL_RECEIPT"
echo "Fresh Java build manifest: $FRESH_JAVA_BUILDS_REAL"
if [[ "$SDX_VARIANT" == "vulkan" ]]; then
    echo "Capability: device-ready Vulkan runtime; an AOT SPIR-V model and physical Vulkan GPU are required"
elif [[ "$DEVICE_READY" == "1" ]]; then
    echo "Capability: device-ready vendor adapter validated"
else
    echo "Capability: runtime contract only; inject the vendor adapter for device execution"
fi
