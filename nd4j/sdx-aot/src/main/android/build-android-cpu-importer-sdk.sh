#!/usr/bin/env bash
# Build and atomically publish the provider-independent Android ARM64 CPU importer SDK.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: build-android-cpu-importer-sdk.sh [options]

Required:
  --android-ndk DIR        Android NDK root
  --java-home DIR          JDK 17 root
  --maven FILE             Maven executable
  --ccache FILE            ccache executable

Optional:
  --android-api N          Android API (default: 28)
  --jobs N                 Native build jobs (default: 12)
  --output-link DIR        Published SDK symlink (default: target/android-cpu-importer)
  --work-dir DIR           Disposable assembly root
  --skip-native-build      Expert override: assemble already-installed Maven artifacts
  --offline                Require Maven offline mode
  -h, --help               Show this help

The default path reuses independently receipted native, managed-runtime, and published
SDK stages. Only the first invalid stage is rebuilt. Native compilation is separated
from Maven packaging so a managed or publication failure never recreates libnd4j's
large native archive. Deployment copies have only their debug sections removed before
they are audited and published as an immutable SDK.
This SDK imports GGUF/GGML into canonical SDZ; it does not execute compiled CPU plans.
Triton and its LLVM/MLIR compiler closure are therefore disabled and forbidden from the
published importer. The separate standalone libsdx_cpu.so is not part of the JavaCPP
importer runtime. Accelerator provider libraries are forbidden.
USAGE
}

fail() {
  printf 'build-android-cpu-importer-sdk: %s\n' "$*" >&2
  exit 3
}

sha256_file() {
  sha256sum "$1" | cut -d ' ' -f 1
}

receipt_has() {
  local receipt="$1"
  local expected="$2"
  [[ -f "$receipt" && ! -L "$receipt" ]] && grep -Fqx -- "$expected" "$receipt"
}

extract_android_native_member() {
  local archive="$1"
  local member="$2"
  local destination_dir="$3"
  local scratch_dir="$4"
  local name candidate existing candidate_sha existing_sha
  name="$(basename -- "$member")"
  [[ "$name" =~ ^lib[A-Za-z0-9._+-]+[.]so$ ]] ||
    fail "unsafe Android native member name: $member"
  candidate="$scratch_dir/candidate-$name"
  unzip -p "$archive" "$member" >"$candidate" ||
    fail "could not extract $member from $archive"
  [[ -s "$candidate" ]] || fail "empty Android native member: $member"
  existing="$destination_dir/$name"
  if [[ -e "$existing" ]]; then
    candidate_sha="$(sha256_file "$candidate")"
    existing_sha="$(sha256_file "$existing")"
    [[ "$candidate_sha" == "$existing_sha" ]] ||
      fail "conflicting Android native libraries named $name"
    rm -f -- "$candidate"
  else
    mv -- "$candidate" "$existing"
  fi
}

validate_native_payload_manifest() {
  local payload_dir="$1"
  local payload_bytes="$2"
  local expected_sha name
  [[ -d "$payload_dir" && ! -L "$payload_dir" && -s "$payload_bytes" ]] || return 1
  cmp -s \
    <(cut -d ' ' -f 2- "$payload_bytes" | LC_ALL=C sort -u) \
    <(find "$payload_dir" -maxdepth 1 -type f -name '*.so' -printf '%f\n' | LC_ALL=C sort -u) || return 1
  while read -r expected_sha name; do
    [[ "$expected_sha" =~ ^[0-9a-f]{64}$ && "$name" =~ ^lib[A-Za-z0-9._+-]+[.]so$ ]] || return 1
    [[ -f "$payload_dir/$name" && ! -L "$payload_dir/$name" && -s "$payload_dir/$name" ]] || return 1
    [[ "$(sha256_file "$payload_dir/$name")" == "$expected_sha" ]] || return 1
  done <"$payload_bytes"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODULE_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
DL4J_ROOT="$(cd "$MODULE_DIR/../.." && pwd)"
SOURCE_MANIFEST_HELPER="$SCRIPT_DIR/source-manifest.sh"
NATIVE_PLATFORM="$DL4J_ROOT/build-scripts/release/native-platform.sh"
[[ -r "$SOURCE_MANIFEST_HELPER" ]] ||
  fail "source manifest helper is missing: $SOURCE_MANIFEST_HELPER"
# shellcheck source=source-manifest.sh
source "$SOURCE_MANIFEST_HELPER"
NATIVE_SOURCE_ROOTS=(
  pom.xml
  build-scripts/release/native-platform.sh
  libnd4j
  # The CPU/NNAPI build excludes Vulkan sources, while CMake regenerates this catalog
  # in-place during configuration. It is build output, not an input to this producer.
  ':(exclude)libnd4j/include/graph/vulkan/VulkanKernelEmitterCatalog.cpp'
)
MANAGED_SOURCE_ROOTS=(
  pom.xml
  nd4j/pom.xml
  nd4j/sdx-aot/pom.xml
  nd4j/sdx-aot/src/main/java
  nd4j/sdx-aot/src/main/resources
  nd4j/sdx-aot/src/main/assembly
  nd4j/sdx-aot/src/main/linker
  nd4j/nd4j-backends/nd4j-api-parent/nd4j-api
  nd4j/nd4j-backends/nd4j-api-parent/nd4j-native-api
  nd4j/nd4j-backends/nd4j-backend-impls/nd4j-presets-common
  nd4j/nd4j-backends/nd4j-backend-impls/nd4j-native-preset
  nd4j/nd4j-backends/nd4j-backend-impls/nd4j-cpu-backend-common
  nd4j/nd4j-backends/nd4j-backend-impls/nd4j-native
  nd4j/nd4j-backends/nd4j-backend-impls/nd4j-sdx
  nd4j/nd4j-backends/nd4j-backend-impls/nd4j-sdx-model
  nd4j/nd4j-backends/nd4j-backend-impls/nd4j-sdx-preset
)
RUNTIME_SOURCE_ROOTS=(
  "${NATIVE_SOURCE_ROOTS[@]}"
  "${MANAGED_SOURCE_ROOTS[@]}"
  nd4j/sdx-aot/src/main/android/build-android-cpu-importer-sdk.sh
  nd4j/sdx-aot/src/main/android/source-manifest.sh
)
ANDROID_NDK=""
JAVA_HOME_ARG=""
MAVEN=""
CCACHE=""
ANDROID_API=28
JOBS=12
OUTPUT_LINK="$MODULE_DIR/target/android-cpu-importer"
WORK_DIR="$MODULE_DIR/target/android-cpu-importer-current-build"
SKIP_NATIVE_BUILD=0
OFFLINE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --android-ndk) ANDROID_NDK="${2:?missing value for --android-ndk}"; shift 2 ;;
    --java-home) JAVA_HOME_ARG="${2:?missing value for --java-home}"; shift 2 ;;
    --maven) MAVEN="${2:?missing value for --maven}"; shift 2 ;;
    --ccache) CCACHE="${2:?missing value for --ccache}"; shift 2 ;;
    --android-api) ANDROID_API="${2:?missing value for --android-api}"; shift 2 ;;
    --jobs) JOBS="${2:?missing value for --jobs}"; shift 2 ;;
    --output-link) OUTPUT_LINK="${2:?missing value for --output-link}"; shift 2 ;;
    --work-dir) WORK_DIR="${2:?missing value for --work-dir}"; shift 2 ;;
    --skip-native-build) SKIP_NATIVE_BUILD=1; shift ;;
    --offline) OFFLINE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) fail "unknown argument: $1" ;;
  esac
done

for required in ANDROID_NDK JAVA_HOME_ARG MAVEN CCACHE; do
  [[ -n "${!required}" ]] || fail "required argument is missing: $required"
done
[[ "$ANDROID_API" =~ ^[1-9][0-9]*$ ]] || fail "Android API must be a positive integer"
[[ "$JOBS" =~ ^[1-9][0-9]*$ ]] || fail "jobs must be a positive integer"
[[ -x "$JAVA_HOME_ARG/bin/java" && -x "$JAVA_HOME_ARG/bin/javac" ]] ||
  fail "JDK is incomplete: $JAVA_HOME_ARG"
[[ -x "$MAVEN" ]] || fail "Maven executable is missing: $MAVEN"
[[ -x "$CCACHE" ]] || fail "ccache executable is missing: $CCACHE"
[[ -x "$NATIVE_PLATFORM" ]] || fail "native platform producer is missing: $NATIVE_PLATFORM"
[[ -s "$ANDROID_NDK/source.properties" ]] || fail "Android NDK is incomplete: $ANDROID_NDK"
NDK_REVISION="$(awk -F= '/Pkg.Revision/ {gsub(/[[:space:]]/, "", $2); print $2}' "$ANDROID_NDK/source.properties")"
[[ "$NDK_REVISION" == 28.1.13356709 ]] ||
  fail "expected Android NDK 28.1.13356709, found $NDK_REVISION"
command -v unzip >/dev/null || fail "unzip is required"
command -v flock >/dev/null || fail "flock is required"

JAVA_SPECIFICATION_VERSION="$("$JAVA_HOME_ARG/bin/java" -XshowSettings:properties -version 2>&1 |
  sed -n 's/^[[:space:]]*java[.]specification[.]version = //p')"
[[ "$JAVA_SPECIFICATION_VERSION" == 17 ]] || fail "JDK 17 is required"
MAVEN="$(realpath -e -- "$MAVEN")"
JAVA_HOME_ARG="$(realpath -e -- "$JAVA_HOME_ARG")"
CCACHE="$(realpath -e -- "$CCACHE")"
ANDROID_NDK="$(realpath -e -- "$ANDROID_NDK")"
OUTPUT_LINK_BASENAME="$(basename -- "$OUTPUT_LINK")"
OUTPUT_PARENT="$(realpath -m -- "$(dirname -- "$OUTPUT_LINK")")"
OUTPUT_LINK="$OUTPUT_PARENT/$OUTPUT_LINK_BASENAME"
WORK_DIR="$(realpath -m -- "$WORK_DIR")"
GENERATIONS_DIR="$OUTPUT_PARENT/.android-cpu-importer-generations"
NATIVE_BUILDS_DIR="$WORK_DIR/native-builds"
MANAGED_STAGES_DIR="$WORK_DIR/managed-stages"
mkdir -p "$WORK_DIR" "$GENERATIONS_DIR" "$NATIVE_BUILDS_DIR" "$MANAGED_STAGES_DIR"

CLEANUP_PATHS=()
cleanup_paths() {
  local path
  for path in "${CLEANUP_PATHS[@]}"; do
    [[ -e "$path" ]] || continue
    chmod -R u+w -- "$path" 2>/dev/null || true
    rm -rf -- "$path"
  done
}
trap cleanup_paths EXIT

write_source_manifest_text() {
  local output="${1:?output path is required}"
  shift
  local relative mode digest
  : >"$output"
  while IFS= read -r -d '' relative &&
    IFS= read -r -d '' mode &&
    IFS= read -r -d '' digest; do
    printf '%s\t%s\t%s\n' "$relative" "$mode" "$digest" >>"$output"
  done < <(sdx_git_source_manifest "$@")
}

SOURCE_MANIFEST_SHA256="$(sdx_git_source_manifest_sha256 "$DL4J_ROOT" "${RUNTIME_SOURCE_ROOTS[@]}")"
SOURCE_MANIFEST_BEFORE="$(mktemp "$WORK_DIR/runtime-source-manifest.before.XXXXXXXX")"
CLEANUP_PATHS+=("$SOURCE_MANIFEST_BEFORE")
write_source_manifest_text "$SOURCE_MANIFEST_BEFORE" "$DL4J_ROOT" "${RUNTIME_SOURCE_ROOTS[@]}"
NATIVE_SOURCE_MANIFEST_SHA256="$(sdx_git_source_manifest_sha256 "$DL4J_ROOT" "${NATIVE_SOURCE_ROOTS[@]}")"
NATIVE_SOURCE_MANIFEST_BEFORE="$(mktemp "$WORK_DIR/native-source-manifest.before.XXXXXXXX")"
CLEANUP_PATHS+=("$NATIVE_SOURCE_MANIFEST_BEFORE")
write_source_manifest_text "$NATIVE_SOURCE_MANIFEST_BEFORE" "$DL4J_ROOT" "${NATIVE_SOURCE_ROOTS[@]}"
MANAGED_SOURCE_MANIFEST_SHA256="$(sdx_git_source_manifest_sha256 "$DL4J_ROOT" "${MANAGED_SOURCE_ROOTS[@]}")"
MANAGED_SOURCE_MANIFEST_BEFORE="$(mktemp "$WORK_DIR/managed-source-manifest.before.XXXXXXXX")"
CLEANUP_PATHS+=("$MANAGED_SOURCE_MANIFEST_BEFORE")
write_source_manifest_text "$MANAGED_SOURCE_MANIFEST_BEFORE" "$DL4J_ROOT" "${MANAGED_SOURCE_ROOTS[@]}"
NDK_SOURCE_PROPERTIES_SHA256="$(sha256_file "$ANDROID_NDK/source.properties")"
JAVA_RELEASE_SHA256="$(sha256_file "$JAVA_HOME_ARG/release")"
MAVEN_ID_SHA256="$({ sha256_file "$MAVEN"; env JAVA_HOME="$JAVA_HOME_ARG" "$MAVEN" --version; } |
  sha256sum | cut -d ' ' -f 1)"
PRODUCER_SHA256="$(sha256_file "${BASH_SOURCE[0]}")"
PROCESS_BLAS_SYMBOLS_ABI=nd4j_process_blas_symbols_abi_v1

validate_published_generation() {
  local generation receipt native_manifest native_bytes expected_sha library_name extra actual_manifest
  [[ -L "$OUTPUT_LINK" ]] || return 1
  generation="$(realpath -e -- "$OUTPUT_LINK")" || return 1
  case "$generation/" in
    "$GENERATIONS_DIR"/*/) ;;
    *) return 1 ;;
  esac
  receipt="$generation/metadata/build-receipt"
  receipt_has "$receipt" "format=2" || return 1
  receipt_has "$receipt" "stage=android-cpu-importer-sdk" || return 1
  receipt_has "$receipt" "cache_schema=independent-stages-v1" || return 1
  receipt_has "$receipt" "source_manifest_sha256=$SOURCE_MANIFEST_SHA256" || return 1
  receipt_has "$receipt" "producer_sha256=$PRODUCER_SHA256" || return 1
  receipt_has "$receipt" "android_api=$ANDROID_API" || return 1
  receipt_has "$receipt" "android_ndk_source_properties_sha256=$NDK_SOURCE_PROPERTIES_SHA256" || return 1
  native_manifest="$generation/metadata/cmake-owned-native-libraries.txt"
  native_bytes="$generation/metadata/native-bytes.txt"
  [[ -s "$native_manifest" && -s "$native_bytes" ]] || return 1
  actual_manifest="$(mktemp "$WORK_DIR/published-native-manifest.XXXXXXXX")"
  CLEANUP_PATHS+=("$actual_manifest")
  find "$generation/jni/arm64-v8a" -maxdepth 1 -type f -name '*.so' -printf '%f\n' |
    LC_ALL=C sort -u >"$actual_manifest"
  cmp -s "$native_manifest" "$actual_manifest" || return 1
  while read -r expected_sha library_name extra; do
    [[ -n "$expected_sha" && -n "$library_name" && -z "${extra:-}" ]] || return 1
    [[ "$library_name" =~ ^lib[A-Za-z0-9._+-]+[.]so$ ]] || return 1
    [[ -f "$generation/jni/arm64-v8a/$library_name" && ! -L "$generation/jni/arm64-v8a/$library_name" ]] || return 1
    [[ "$(sha256_file "$generation/jni/arm64-v8a/$library_name")" == "$expected_sha" ]] || return 1
  done <"$native_bytes"
  PUBLISHED_GENERATION="$generation"
}

if validate_published_generation; then
  printf 'Reusing validated Android CPU importer SDK: %s\n' "$PUBLISHED_GENERATION"
  exit 0
fi

run_native_platform_stage() {
  local goal="$1"
  shift
  local maven_flags=("$@")
  local native_only=0
  [[ "$goal" != compile ]] || native_only=1
  [[ "$OFFLINE" == 0 ]] || maven_flags=(-o "${maven_flags[@]}")
  env \
    DL4J_FAMILY=android-arm64 \
    DL4J_NATIVE_ONLY="$native_only" \
    DL4J_MAVEN_ALSO_MAKE=0 \
    DL4J_BUILD_THREADS="$JOBS" \
    DL4J_MAVEN_GOAL="$goal" \
    DL4J_MVN_FLAGS="${maven_flags[*]}" \
    DL4J_CMAKE_ARGS="-DBLAS_IMPL=openblas" \
    DL4J_ANDROID_API="$ANDROID_API" \
    DL4J_COMPILER_CACHE="$CCACHE" \
    DL4J_NATIVE_OUTPUT_ROOT="$NATIVE_OUTPUT_ROOT" \
    ANDROID_NDK="$ANDROID_NDK" \
    JAVA_HOME="$JAVA_HOME_ARG" \
    PATH="$(dirname "$MAVEN"):$(dirname "$CCACHE"):$JAVA_HOME_ARG/bin:$PATH" \
    "$NATIVE_PLATFORM" --run
}

NATIVE_STAGE_KEY="$({
  printf '%s\n' \
    'format=android-cpu-native-stage-v2' \
    "source=$NATIVE_SOURCE_MANIFEST_SHA256" \
    "ndk=$NDK_SOURCE_PROPERTIES_SHA256" \
    "android_api=$ANDROID_API" \
    'android_abi=arm64-v8a' \
    'blas=openblas' \
    'triton=off' \
    'sdx_standalone=on'
} | sha256sum | cut -d ' ' -f 1)"

# Native compilation always uses one directly addressed workspace. Immutable
# publication stages remain content-addressed, but CMake dependency state is not
# wrapped in pointers, compatibility hashes, or generation directories.
NATIVE_OUTPUT_ROOT="$NATIVE_BUILDS_DIR/current"
NATIVE_OUTPUT_ROOT="$(realpath -m -- "$NATIVE_OUTPUT_ROOT")"
[[ "$NATIVE_OUTPUT_ROOT" == "$(realpath -m -- "$NATIVE_BUILDS_DIR/current")" ]] ||
  fail "native workspace did not resolve to the canonical current directory"
[[ ! -e "$NATIVE_OUTPUT_ROOT" || ( -d "$NATIVE_OUTPUT_ROOT" && ! -L "$NATIVE_OUTPUT_ROOT" ) ]] ||
  fail "unsafe native workspace root: $NATIVE_OUTPUT_ROOT"
mkdir -p "$NATIVE_OUTPUT_ROOT"
NATIVE_OUTPUT_ROOT="$(realpath -e -- "$NATIVE_OUTPUT_ROOT")"
exec {NATIVE_BUILD_LOCK_FD}>"$NATIVE_BUILDS_DIR/build.lock"
flock "$NATIVE_BUILD_LOCK_FD"

NATIVE_BUILD_DIR="$NATIVE_OUTPUT_ROOT/android-arm64-api${ANDROID_API}-cpu"
NATIVE_CPU_BACKEND="$NATIVE_BUILD_DIR/libnd4jcpu.so"
NATIVE_STAGE_RECEIPT="$NATIVE_BUILD_DIR/android-cpu-native-stage.receipt"
NATIVE_NM="$ANDROID_NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-nm"
[[ -x "$NATIVE_NM" ]] || fail "NDK llvm-nm is missing"

validate_native_stage() {
  [[ -s "$NATIVE_BUILD_DIR/CMakeCache.txt" && -s "$NATIVE_CPU_BACKEND" ]] || return 1
  receipt_has "$NATIVE_STAGE_RECEIPT" "format=3" || return 1
  receipt_has "$NATIVE_STAGE_RECEIPT" "native_workspace=stable-current-v1" || return 1
  receipt_has "$NATIVE_STAGE_RECEIPT" "stage=android-cpu-native" || return 1
  receipt_has "$NATIVE_STAGE_RECEIPT" "stage_key=$NATIVE_STAGE_KEY" || return 1
  receipt_has "$NATIVE_STAGE_RECEIPT" "source_manifest_sha256=$NATIVE_SOURCE_MANIFEST_SHA256" || return 1
  receipt_has "$NATIVE_STAGE_RECEIPT" "android_ndk_source_properties_sha256=$NDK_SOURCE_PROPERTIES_SHA256" || return 1
  receipt_has "$NATIVE_STAGE_RECEIPT" "native_cpu_sha256=$(sha256_file "$NATIVE_CPU_BACKEND")" || return 1
  "$NATIVE_NM" -D --defined-only "$NATIVE_CPU_BACKEND" |
    grep -E "[[:space:]]$PROCESS_BLAS_SYMBOLS_ABI$" >/dev/null
}

if validate_native_stage; then
  printf 'Reusing validated native compile stage: %s\n' "$NATIVE_STAGE_KEY"
else
  if [[ "$SKIP_NATIVE_BUILD" == 0 ]]; then
    if [[ -s "$NATIVE_BUILD_DIR/CMakeCache.txt" ]]; then
      printf 'Incrementally updating stable native CMake workspace: %s\n' "$NATIVE_OUTPUT_ROOT"
    else
      printf 'Initializing stable native CMake workspace: %s\n' "$NATIVE_OUTPUT_ROOT"
    fi
    run_native_platform_stage compile -Dlibnd4j.triton=OFF
  else
    printf 'Expert override: validating existing native output without compiling it.\n'
  fi
  [[ -s "$NATIVE_BUILD_DIR/CMakeCache.txt" && -s "$NATIVE_CPU_BACKEND" ]] ||
    fail "canonical native producer did not leave a complete Android CPU build"
  "$NATIVE_NM" -D --defined-only "$NATIVE_CPU_BACKEND" |
    grep -E "[[:space:]]$PROCESS_BLAS_SYMBOLS_ABI$" >/dev/null ||
    fail "native CPU build lacks process BLAS symbol-resolution ABI"
  current_native_source_manifest_sha256="$(sdx_git_source_manifest_sha256 "$DL4J_ROOT" "${NATIVE_SOURCE_ROOTS[@]}")"
  if [[ "$current_native_source_manifest_sha256" != "$NATIVE_SOURCE_MANIFEST_SHA256" ]]; then
    native_source_manifest_after="$(mktemp "$WORK_DIR/native-source-manifest.after.XXXXXXXX")"
    CLEANUP_PATHS+=("$native_source_manifest_after")
    write_source_manifest_text "$native_source_manifest_after" "$DL4J_ROOT" "${NATIVE_SOURCE_ROOTS[@]}"
    printf 'Native source manifest changed during compilation:\n' >&2
    diff -u "$NATIVE_SOURCE_MANIFEST_BEFORE" "$native_source_manifest_after" >&2 || true
    fail "native sources changed during the Android CPU compile stage"
  fi
  native_receipt_tmp="$NATIVE_STAGE_RECEIPT.tmp.$$"
  cat >"$native_receipt_tmp" <<RECEIPT
format=3
stage=android-cpu-native
stage_key=$NATIVE_STAGE_KEY
source_manifest_sha256=$NATIVE_SOURCE_MANIFEST_SHA256
native_workspace=stable-current-v1
android_api=$ANDROID_API
android_abi=arm64-v8a
android_ndk_source_properties_sha256=$NDK_SOURCE_PROPERTIES_SHA256
native_cpu_sha256=$(sha256_file "$NATIVE_CPU_BACKEND")
process_blas_symbols_capability=$PROCESS_BLAS_SYMBOLS_ABI
RECEIPT
  mv -f -- "$native_receipt_tmp" "$NATIVE_STAGE_RECEIPT"
fi
NATIVE_CPU_SHA256="$(sha256_file "$NATIVE_CPU_BACKEND")"

MANAGED_STAGE_KEY="$({
  printf '%s\n' \
    'format=android-cpu-managed-stage-v2' \
    "source=$MANAGED_SOURCE_MANIFEST_SHA256" \
    "native_stage=$NATIVE_STAGE_KEY" \
    "native_cpu=$NATIVE_CPU_SHA256" \
    "java=$JAVA_RELEASE_SHA256" \
    "maven=$MAVEN_ID_SHA256" \
    'javacpp_platform=android-arm64' \
    'libnd4j_archive=skipped' \
    'payload=immutable-importer-closure-v2-no-compiler'
} | sha256sum | cut -d ' ' -f 1)"
MANAGED_STAGE_DIR="$MANAGED_STAGES_DIR/$MANAGED_STAGE_KEY"
MANAGED_STAGE_RECEIPT="$MANAGED_STAGE_DIR/managed-stage.receipt"
MANAGED_NATIVE_PAYLOAD="$MANAGED_STAGE_DIR/native-libraries"
MANAGED_NATIVE_BYTES="$MANAGED_STAGE_DIR/native-library-bytes.txt"

validate_managed_stage() {
  [[ -d "$MANAGED_STAGE_DIR" && ! -L "$MANAGED_STAGE_DIR" ]] || return 1
  receipt_has "$MANAGED_STAGE_RECEIPT" "format=2" || return 1
  receipt_has "$MANAGED_STAGE_RECEIPT" "stage=android-cpu-managed-runtime" || return 1
  receipt_has "$MANAGED_STAGE_RECEIPT" "stage_key=$MANAGED_STAGE_KEY" || return 1
  receipt_has "$MANAGED_STAGE_RECEIPT" "source_manifest_sha256=$MANAGED_SOURCE_MANIFEST_SHA256" || return 1
  receipt_has "$MANAGED_STAGE_RECEIPT" "native_stage_key=$NATIVE_STAGE_KEY" || return 1
  [[ -s "$MANAGED_STAGE_DIR/runtime-classpath.txt" &&
     -s "$MANAGED_STAGE_DIR/classpath.entries" &&
     -s "$MANAGED_STAGE_DIR/classpath-bytes.txt" &&
     -s "$MANAGED_NATIVE_BYTES" ]] || return 1
  receipt_has "$MANAGED_STAGE_RECEIPT" "classpath_bytes_sha256=$(sha256_file "$MANAGED_STAGE_DIR/classpath-bytes.txt")" || return 1
  receipt_has "$MANAGED_STAGE_RECEIPT" "native_payload_bytes_sha256=$(sha256_file "$MANAGED_NATIVE_BYTES")" || return 1
  validate_native_payload_manifest "$MANAGED_NATIVE_PAYLOAD" "$MANAGED_NATIVE_BYTES"
}

if validate_managed_stage; then
  printf 'Reusing validated managed runtime stage: %s\n' "$MANAGED_STAGE_KEY"
else
  printf 'Managed runtime cache miss: %s\n' "$MANAGED_STAGE_KEY"
  if [[ "$SKIP_NATIVE_BUILD" == 0 ]]; then
    # The native compile is already receipted. This install packages that exact output
    # without running CMake or generating libnd4j's multi-gigabyte assembly archive.
    run_native_platform_stage install \
      -Dlibnd4j.triton=OFF \
      -Dlibnd4j.native.compile.skip=true \
      -Dassembly.skipAssembly=true
  else
    printf 'Expert override: validating already-installed managed artifacts.\n'
  fi
  managed_tmp="$(mktemp -d "$WORK_DIR/managed-stage.XXXXXXXX")"
  CLEANUP_PATHS+=("$managed_tmp")
  classpath_flags=(-DskipTests -Djavacpp.platform=android-arm64)
  [[ "$OFFLINE" == 0 ]] || classpath_flags+=(-o)
  env JAVA_HOME="$JAVA_HOME_ARG" PATH="$JAVA_HOME_ARG/bin:$PATH" \
    "$MAVEN" "${classpath_flags[@]}" -f "$MODULE_DIR/pom.xml" \
      dependency:build-classpath -Dmdep.includeScope=runtime \
      -Dmdep.outputFile="$managed_tmp/runtime-classpath.txt"
  [[ -s "$managed_tmp/runtime-classpath.txt" ]] || fail "Maven produced no Android runtime classpath"
  tr ':' '\n' <"$managed_tmp/runtime-classpath.txt" |
    sed '/^[[:space:]]*$/d' >"$managed_tmp/classpath.entries"
  : >"$managed_tmp/classpath-bytes.txt"
  mkdir -p "$managed_tmp/native-libraries"
  archive_index=0
  while IFS= read -r archive; do
    [[ -f "$archive" && ! -L "$archive" && -s "$archive" ]] ||
      fail "unsafe runtime classpath artifact: $archive"
    printf '%s %s\n' "$(sha256_file "$archive")" "$(realpath -e -- "$archive")" >>"$managed_tmp/classpath-bytes.txt"
    members="$managed_tmp/members.$archive_index"
    archive_index=$((archive_index + 1))
    unzip -Z1 "$archive" |
      grep -E '(^|/)(android-arm64|arm64-v8a)/lib[A-Za-z0-9._+-]+[.]so$' |
      LC_ALL=C sort -u >"$members" || true
    while IFS= read -r member; do
      [[ -n "$member" ]] || continue
      # The standalone C runtime is published directly from the receipted native
      # stage. It is intentionally absent from the JavaCPP importer closure, so
      # do not inflate its multi-gigabyte ZIP64 member just to delete it below.
      case "$(basename -- "$member")" in
        libsdx_cpu.so|libLLVM.so|libMLIR.so) continue ;;
      esac
      extract_android_native_member "$archive" "$member" "$managed_tmp/native-libraries" "$managed_tmp"
    done <"$members"
    rm -f -- "$members"
  done <"$managed_tmp/classpath.entries"
  LC_ALL=C sort -o "$managed_tmp/classpath-bytes.txt" "$managed_tmp/classpath-bytes.txt"

  shopt -s nullglob
  ndk_libomp_candidates=(
    "$ANDROID_NDK"/toolchains/llvm/prebuilt/linux-x86_64/lib/clang/*/lib/linux/aarch64/libomp.so
  )
  shopt -u nullglob
  [[ "${#ndk_libomp_candidates[@]}" == 1 ]] ||
    fail "expected exactly one AArch64 NDK libomp.so, found ${#ndk_libomp_candidates[@]}"
  ndk_libomp="${ndk_libomp_candidates[0]}"
  if [[ -e "$managed_tmp/native-libraries/libomp.so" ]]; then
    [[ "$(sha256_file "$managed_tmp/native-libraries/libomp.so")" == "$(sha256_file "$ndk_libomp")" ]] ||
      fail "Maven and Android NDK supplied conflicting libomp.so bytes"
  else
    cp -- "$ndk_libomp" "$managed_tmp/native-libraries/libomp.so"
  fi
  rm -f -- "$managed_tmp/native-libraries/libsdx_cpu.so"
  for required_library in \
    libjnind4jcpu.so \
    libnd4jcpu.so \
    libopenblas.so \
    libomp.so \
    libjnitokenizers.so \
    libtokenizers_ffi.so \
    libtokenizers_wrapper.so; do
    [[ -f "$managed_tmp/native-libraries/$required_library" &&
       ! -L "$managed_tmp/native-libraries/$required_library" &&
       -s "$managed_tmp/native-libraries/$required_library" ]] ||
      fail "resolved Android CPU importer closure omitted $required_library"
  done
  for provider_library in libnd4jnnapi.so libnd4jvulkan.so liblitert-lm.so; do
    [[ ! -e "$managed_tmp/native-libraries/$provider_library" ]] ||
      fail "CPU importer closure contains accelerator provider $provider_library"
  done
  for compiler_library in libLLVM.so libMLIR.so; do
    [[ ! -e "$managed_tmp/native-libraries/$compiler_library" ]] ||
      fail "CPU importer closure contains unused compiler runtime $compiler_library"
  done
  : >"$managed_tmp/native-library-bytes.txt"
  while IFS= read -r library_name; do
    printf '%s %s\n' \
      "$(sha256_file "$managed_tmp/native-libraries/$library_name")" \
      "$library_name"
  done < <(find "$managed_tmp/native-libraries" -maxdepth 1 -type f -name '*.so' -printf '%f\n' | LC_ALL=C sort -u) \
    >"$managed_tmp/native-library-bytes.txt"
  validate_native_payload_manifest "$managed_tmp/native-libraries" "$managed_tmp/native-library-bytes.txt" ||
    fail "could not validate immutable managed native payload"
  current_managed_source_manifest_sha256="$(sdx_git_source_manifest_sha256 "$DL4J_ROOT" "${MANAGED_SOURCE_ROOTS[@]}")"
  if [[ "$current_managed_source_manifest_sha256" != "$MANAGED_SOURCE_MANIFEST_SHA256" ]]; then
    managed_source_manifest_after="$(mktemp "$WORK_DIR/managed-source-manifest.after.XXXXXXXX")"
    CLEANUP_PATHS+=("$managed_source_manifest_after")
    write_source_manifest_text "$managed_source_manifest_after" "$DL4J_ROOT" "${MANAGED_SOURCE_ROOTS[@]}"
    printf "Managed source manifest changed during the Android CPU runtime stage:\\n" >&2
    diff -u "$MANAGED_SOURCE_MANIFEST_BEFORE" "$managed_source_manifest_after" >&2 || true
    fail "managed sources changed during the Android CPU runtime stage"
  fi
  cat >"$managed_tmp/managed-stage.receipt" <<RECEIPT
format=2
stage=android-cpu-managed-runtime
stage_key=$MANAGED_STAGE_KEY
source_manifest_sha256=$MANAGED_SOURCE_MANIFEST_SHA256
native_stage_key=$NATIVE_STAGE_KEY
native_cpu_sha256=$NATIVE_CPU_SHA256
java_release_sha256=$JAVA_RELEASE_SHA256
maven_id_sha256=$MAVEN_ID_SHA256
classpath_bytes_sha256=$(sha256_file "$managed_tmp/classpath-bytes.txt")
native_payload_bytes_sha256=$(sha256_file "$managed_tmp/native-library-bytes.txt")
RECEIPT
  if [[ -e "$MANAGED_STAGE_DIR" ]]; then
    chmod -R u+w -- "$MANAGED_STAGE_DIR" 2>/dev/null || true
    rm -rf -- "$MANAGED_STAGE_DIR"
  fi
  mv -- "$managed_tmp" "$MANAGED_STAGE_DIR"
  chmod -R a-w -- "$MANAGED_STAGE_DIR"
fi

BUILD_ROOT="$(mktemp -d "$WORK_DIR/generation.XXXXXXXX")"
CLEANUP_PATHS+=("$BUILD_ROOT")
STAGE="$BUILD_ROOT/sdk"
JNI_DIR="$STAGE/jni/arm64-v8a"
METADATA_DIR="$STAGE/metadata"
mkdir -p "$JNI_DIR" "$METADATA_DIR"
cp -- "$MANAGED_STAGE_DIR/classpath-bytes.txt" "$METADATA_DIR/classpath-bytes.txt"
CLASSPATH_BYTES="$METADATA_DIR/classpath-bytes.txt"
cp -- "$MANAGED_NATIVE_PAYLOAD"/*.so "$JNI_DIR/"

# The managed stage snapshots the complete native closure before Maven's mutable local
# repository can be changed by another build. Publication now depends only on immutable,
# content-verified stage files rather than reopening gigabyte classifier JARs.
validate_native_payload_manifest "$MANAGED_NATIVE_PAYLOAD" "$MANAGED_NATIVE_BYTES" ||
  fail "managed native payload changed during Android CPU importer publication"

TOOLCHAIN="$ANDROID_NDK/toolchains/llvm/prebuilt/linux-x86_64/bin"
READELF="$TOOLCHAIN/llvm-readelf"
LLVM_NM="$TOOLCHAIN/llvm-nm"
LLVM_STRIP="$TOOLCHAIN/llvm-strip"
[[ -x "$READELF" ]] || fail "NDK llvm-readelf is missing"
[[ -x "$LLVM_NM" ]] || fail "NDK llvm-nm is missing"
[[ -x "$LLVM_STRIP" ]] || fail "NDK llvm-strip is missing"

# CMake and Maven artifacts retain their full symbols and are bound by classpath-bytes.txt.
# APK deployment copies need executable code, unwind information, build IDs, and dynamic
# symbols—not multi-gigabyte DWARF sections. --strip-debug preserves the load-time ABI.
for library in "$JNI_DIR"/*.so; do
  "$LLVM_STRIP" --strip-debug "$library" ||
    fail "could not strip debug sections from deployment library: $(basename -- "$library")"
done

for required_library in \
  libjnind4jcpu.so \
  libnd4jcpu.so \
  libopenblas.so \
  libomp.so \
  libjnitokenizers.so \
  libtokenizers_ffi.so \
  libtokenizers_wrapper.so; do
  [[ -f "$JNI_DIR/$required_library" && ! -L "$JNI_DIR/$required_library" &&
     -s "$JNI_DIR/$required_library" ]] ||
    fail "resolved Android CPU importer closure omitted $required_library"
done
for provider_library in libnd4jnnapi.so libnd4jvulkan.so liblitert-lm.so; do
  [[ ! -e "$JNI_DIR/$provider_library" ]] ||
    fail "CPU importer closure contains accelerator provider $provider_library"
done
for compiler_library in libLLVM.so libMLIR.so; do
  [[ ! -e "$JNI_DIR/$compiler_library" ]] ||
    fail "CPU importer deployment contains unused compiler runtime $compiler_library"
done

is_android_system_library() {
  case "$1" in
    libc.so|libdl.so|libm.so|liblog.so|libz.so|libandroid.so|libatomic.so|libc++_shared.so|libneuralnetworks.so) return 0 ;;
    *) return 1 ;;
  esac
}
for library in "$JNI_DIR"/*.so; do
  name="$(basename -- "$library")"
  elf_header="$BUILD_ROOT/$name.header"
  "$READELF" -h "$library" >"$elf_header" || fail "ELF header audit failed: $name"
  grep -Eq 'Class:[[:space:]]+ELF64' "$elf_header" || fail "$name is not ELF64"
  grep -Eq 'Machine:[[:space:]]+AArch64' "$elf_header" || fail "$name is not AArch64"
  if "$READELF" --sections "$library" | grep -Eq '[.]debug_'; then
    fail "$name still contains debug sections after deployment stripping"
  fi
  while IFS= read -r needed; do
    is_android_system_library "$needed" || [[ -s "$JNI_DIR/$needed" ]] ||
      fail "$name requires unpackaged native dependency $needed"
  done < <("$READELF" -d "$library" |
    sed -n 's/.*Shared library: \[\([^]]*\)\].*/\1/p' | LC_ALL=C sort -u)
done

CPU_BACKEND="$JNI_DIR/libnd4jcpu.so"
CPU_BACKEND_SYMBOLS="$BUILD_ROOT/libnd4jcpu.dynamic-symbols"
"$LLVM_NM" -D --defined-only "$CPU_BACKEND" >"$CPU_BACKEND_SYMBOLS" ||
  fail "could not inspect CPU importer backend symbols"
grep -q "[[:space:]]$PROCESS_BLAS_SYMBOLS_ABI$" "$CPU_BACKEND_SYMBOLS" ||
  fail "CPU importer backend lacks process BLAS symbol-resolution ABI: $PROCESS_BLAS_SYMBOLS_ABI"
TEXT_GENERATION_V2_CONTRACTS=(
  causal-lm-in-graph-state-v2
  io.recurrentStates
  "duplicate recurrent state input"
)
for contract in "${TEXT_GENERATION_V2_CONTRACTS[@]}"; do
  LC_ALL=C grep -aFq "$contract" "$CPU_BACKEND" ||
    fail "CPU importer backend lacks current text-generation contract: $contract"
done
if LC_ALL=C grep -aFq "executeSegmentWithCpuGraph: no CPU graph backends available" "$CPU_BACKEND"; then
  fail "CPU importer backend contains obsolete missing-backend execution path"
fi

NATIVE_MANIFEST="$METADATA_DIR/cmake-owned-native-libraries.txt"
find "$JNI_DIR" -maxdepth 1 -type f -name '*.so' -printf '%f\n' |
  LC_ALL=C sort -u >"$NATIVE_MANIFEST"
NATIVE_BYTES="$METADATA_DIR/native-bytes.txt"
while IFS= read -r library_name; do
  printf '%s %s\n' "$(sha256_file "$JNI_DIR/$library_name")" "$library_name"
done <"$NATIVE_MANIFEST" >"$NATIVE_BYTES"

current_source_manifest_sha256="$(sdx_git_source_manifest_sha256 "$DL4J_ROOT" "${RUNTIME_SOURCE_ROOTS[@]}")"
if [[ "$current_source_manifest_sha256" != "$SOURCE_MANIFEST_SHA256" ]]; then
  source_manifest_after="$(mktemp "$WORK_DIR/runtime-source-manifest.after.XXXXXXXX")"
  CLEANUP_PATHS+=("$source_manifest_after")
  write_source_manifest_text "$source_manifest_after" "$DL4J_ROOT" "${RUNTIME_SOURCE_ROOTS[@]}"
  printf 'Runtime source manifest changed during the importer build:\n' >&2
  diff -u "$SOURCE_MANIFEST_BEFORE" "$source_manifest_after" >&2 || true
  fail "runtime sources changed during the Android CPU importer build"
fi
INPUTS_SHA256="$({
  printf '%s\n' "$PRODUCER_SHA256"
  sha256_file "$CLASSPATH_BYTES"
  sha256_file "$NATIVE_BYTES"
  printf '%s\n' \
    "$NDK_SOURCE_PROPERTIES_SHA256" \
    "$SOURCE_MANIFEST_SHA256" \
    "$NATIVE_STAGE_KEY" \
    "$MANAGED_STAGE_KEY" \
    "$PROCESS_BLAS_SYMBOLS_ABI" \
    "$ANDROID_API"
} | sha256sum | cut -d ' ' -f 1)"
cat >"$METADATA_DIR/build-receipt" <<RECEIPT
format=2
stage=android-cpu-importer-sdk
cache_schema=independent-stages-v1
inputs_sha256=$INPUTS_SHA256
source_manifest_sha256=$SOURCE_MANIFEST_SHA256
native_source_manifest_sha256=$NATIVE_SOURCE_MANIFEST_SHA256
managed_source_manifest_sha256=$MANAGED_SOURCE_MANIFEST_SHA256
native_stage_key=$NATIVE_STAGE_KEY
managed_stage_key=$MANAGED_STAGE_KEY
producer=$(realpath -e -- "${BASH_SOURCE[0]}")
producer_sha256=$PRODUCER_SHA256
android_api=$ANDROID_API
android_abi=arm64-v8a
process_blas_symbols_abi=1
process_blas_symbols_capability=$PROCESS_BLAS_SYMBOLS_ABI
native_packaging=strip-debug
triton_cpu_included=false
compiler_runtime_included=false
standalone_sdx_cpu_included=false
android_ndk_source_properties_sha256=$NDK_SOURCE_PROPERTIES_SHA256
classpath_bytes_sha256=$(sha256_file "$CLASSPATH_BYTES")
native_bytes_sha256=$(sha256_file "$NATIVE_BYTES")
native_library_count=$(wc -l <"$NATIVE_MANIFEST")
RECEIPT

GENERATION_ID="$SOURCE_MANIFEST_SHA256-$INPUTS_SHA256"
GENERATION="$GENERATIONS_DIR/$GENERATION_ID"
if [[ -e "$GENERATION" ]]; then
  [[ -d "$GENERATION" && ! -L "$GENERATION" ]] || fail "unsafe existing generation: $GENERATION"
  receipt_has "$GENERATION/metadata/build-receipt" "inputs_sha256=$INPUTS_SHA256" ||
    fail "existing Android CPU importer generation has a conflicting receipt"
  rm -rf -- "$STAGE"
else
  mv -- "$STAGE" "$GENERATION"
  chmod -R a-w "$GENERATION"
fi
LINK_TMP="$OUTPUT_LINK.tmp.$$"
CLEANUP_PATHS+=("$LINK_TMP")
ln -s "$GENERATION" "$LINK_TMP"
mv -Tf -- "$LINK_TMP" "$OUTPUT_LINK"
printf 'Published Android CPU importer SDK: %s\n' "$GENERATION"
printf 'Public SDK link: %s\n' "$OUTPUT_LINK"
printf 'Native libraries: %s\n' "$(wc -l <"$GENERATION/metadata/cmake-owned-native-libraries.txt")"
