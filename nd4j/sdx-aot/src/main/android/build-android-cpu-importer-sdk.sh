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
  --skip-native-build      Assemble already-installed current Maven artifacts
  --offline                Require Maven offline mode
  -h, --help               Show this help

The default path rebuilds nd4j-native for Android ARM64 in an API-specific CMake
directory, resolves the Android classified runtime closure from Maven, audits it,
and publishes an immutable SDK. Accelerator provider libraries are forbidden.
USAGE
}

fail() {
  printf 'build-android-cpu-importer-sdk: %s\n' "$*" >&2
  exit 3
}

sha256_file() {
  sha256sum "$1" | cut -d ' ' -f 1
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODULE_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
DL4J_ROOT="$(cd "$MODULE_DIR/../.." && pwd)"
NATIVE_PLATFORM="$DL4J_ROOT/build-scripts/release/native-platform.sh"
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

JAVA_SPECIFICATION_VERSION="$("$JAVA_HOME_ARG/bin/java" -XshowSettings:properties -version 2>&1 |
  sed -n 's/^[[:space:]]*java[.]specification[.]version = //p')"
[[ "$JAVA_SPECIFICATION_VERSION" == 17 ]] || fail "JDK 17 is required"
MAVEN="$(realpath -e -- "$MAVEN")"
JAVA_HOME_ARG="$(realpath -e -- "$JAVA_HOME_ARG")"
CCACHE="$(realpath -e -- "$CCACHE")"
ANDROID_NDK="$(realpath -e -- "$ANDROID_NDK")"

if [[ "$SKIP_NATIVE_BUILD" == 0 ]]; then
  maven_flags=(-Dlibnd4j.triton=ON)
  [[ "$OFFLINE" == 0 ]] || maven_flags=(-o "${maven_flags[@]}")
  env \
    DL4J_FAMILY=android-arm64 \
    DL4J_BUILD_THREADS="$JOBS" \
    DL4J_MAVEN_GOAL=install \
    DL4J_MVN_FLAGS="${maven_flags[*]}" \
    DL4J_CMAKE_ARGS="-DBLAS_IMPL=openblas" \
    DL4J_ANDROID_API="$ANDROID_API" \
    DL4J_COMPILER_CACHE="$CCACHE" \
    ANDROID_NDK="$ANDROID_NDK" \
    JAVA_HOME="$JAVA_HOME_ARG" \
    PATH="$(dirname "$MAVEN"):$(dirname "$CCACHE"):$JAVA_HOME_ARG/bin:$PATH" \
    "$NATIVE_PLATFORM" --run
fi

OUTPUT_PARENT="$(dirname "$OUTPUT_LINK")"
GENERATIONS_DIR="$OUTPUT_PARENT/.android-cpu-importer-generations"
mkdir -p "$WORK_DIR" "$GENERATIONS_DIR"
BUILD_ROOT="$(mktemp -d "$WORK_DIR/generation.XXXXXXXX")"
trap 'chmod -R u+w -- "$BUILD_ROOT" 2>/dev/null || true; rm -rf -- "$BUILD_ROOT"' EXIT
STAGE="$BUILD_ROOT/sdk"
JNI_DIR="$STAGE/jni/arm64-v8a"
METADATA_DIR="$STAGE/metadata"
mkdir -p "$JNI_DIR" "$METADATA_DIR"

classpath_flags=(-DskipTests -Djavacpp.platform=android-arm64)
[[ "$OFFLINE" == 0 ]] || classpath_flags+=(-o)
CLASSPATH_FILE="$BUILD_ROOT/runtime-classpath.txt"
env JAVA_HOME="$JAVA_HOME_ARG" PATH="$JAVA_HOME_ARG/bin:$PATH" \
  "$MAVEN" "${classpath_flags[@]}" -f "$MODULE_DIR/pom.xml" \
    dependency:build-classpath -Dmdep.includeScope=runtime \
    -Dmdep.outputFile="$CLASSPATH_FILE"
[[ -s "$CLASSPATH_FILE" ]] || fail "Maven produced no Android runtime classpath"

tr ':' '\n' <"$CLASSPATH_FILE" | sed '/^[[:space:]]*$/d' >"$BUILD_ROOT/classpath.entries"
CLASSPATH_BYTES="$METADATA_DIR/classpath-bytes.txt"
: >"$CLASSPATH_BYTES"
while IFS= read -r archive; do
  [[ -f "$archive" && ! -L "$archive" && -s "$archive" ]] ||
    fail "unsafe runtime classpath artifact: $archive"
  printf '%s %s\n' "$(sha256_file "$archive")" "$(realpath -e -- "$archive")" >>"$CLASSPATH_BYTES"
done <"$BUILD_ROOT/classpath.entries"
LC_ALL=C sort -o "$CLASSPATH_BYTES" "$CLASSPATH_BYTES"

extract_native_member() {
  local archive="$1"
  local member="$2"
  local name candidate existing candidate_sha existing_sha
  name="$(basename -- "$member")"
  [[ "$name" =~ ^lib[A-Za-z0-9._+-]+[.]so$ ]] ||
    fail "unsafe Android native member name: $member"
  candidate="$BUILD_ROOT/candidate-$name"
  unzip -p "$archive" "$member" >"$candidate" ||
    fail "could not extract $member from $archive"
  [[ -s "$candidate" ]] || fail "empty Android native member: $member"
  existing="$JNI_DIR/$name"
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

while IFS= read -r archive; do
  members="$BUILD_ROOT/members.$(sha256_file "$archive")"
  unzip -Z1 "$archive" |
    grep -E '(^|/)(android-arm64|arm64-v8a)/lib[A-Za-z0-9._+-]+[.]so$' |
    LC_ALL=C sort -u >"$members" || true
  while IFS= read -r member; do
    [[ -n "$member" ]] || continue
    extract_native_member "$archive" "$member"
  done <"$members"
done <"$BUILD_ROOT/classpath.entries"

shopt -s nullglob
ndk_libomp_candidates=(
  "$ANDROID_NDK"/toolchains/llvm/prebuilt/linux-x86_64/lib/clang/*/lib/linux/aarch64/libomp.so
)
shopt -u nullglob
[[ "${#ndk_libomp_candidates[@]}" == 1 ]] ||
  fail "expected exactly one AArch64 NDK libomp.so, found ${#ndk_libomp_candidates[@]}"
NDK_LIBOMP="${ndk_libomp_candidates[0]}"
if [[ -e "$JNI_DIR/libomp.so" ]]; then
  [[ "$(sha256_file "$JNI_DIR/libomp.so")" == "$(sha256_file "$NDK_LIBOMP")" ]] ||
    fail "Maven and Android NDK supplied conflicting libomp.so bytes"
else
  cp -- "$NDK_LIBOMP" "$JNI_DIR/libomp.so"
fi

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

TOOLCHAIN="$ANDROID_NDK/toolchains/llvm/prebuilt/linux-x86_64/bin"
READELF="$TOOLCHAIN/llvm-readelf"
[[ -x "$READELF" ]] || fail "NDK llvm-readelf is missing"
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
  while IFS= read -r needed; do
    is_android_system_library "$needed" || [[ -s "$JNI_DIR/$needed" ]] ||
      fail "$name requires unpackaged native dependency $needed"
  done < <("$READELF" -d "$library" |
    sed -n 's/.*Shared library: \[\([^]]*\)\].*/\1/p' | LC_ALL=C sort -u)
done

CPU_BACKEND="$JNI_DIR/libnd4jcpu.so"
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

INPUTS_SHA256="$({
  sha256_file "${BASH_SOURCE[0]}"
  sha256_file "$CLASSPATH_BYTES"
  sha256_file "$NATIVE_BYTES"
  sha256_file "$ANDROID_NDK/source.properties"
  printf '%s\n' "$ANDROID_API"
} | sha256sum | cut -d ' ' -f 1)"
cat >"$METADATA_DIR/build-receipt" <<RECEIPT
format=1
stage=android-cpu-importer-sdk
inputs_sha256=$INPUTS_SHA256
producer=$(realpath -e -- "${BASH_SOURCE[0]}")
producer_sha256=$(sha256_file "${BASH_SOURCE[0]}")
android_api=$ANDROID_API
android_abi=arm64-v8a
android_ndk_source_properties_sha256=$(sha256_file "$ANDROID_NDK/source.properties")
classpath_bytes_sha256=$(sha256_file "$CLASSPATH_BYTES")
native_bytes_sha256=$(sha256_file "$NATIVE_BYTES")
native_library_count=$(wc -l <"$NATIVE_MANIFEST")
RECEIPT

GENERATION_ID="$(sha256_file "$NATIVE_BYTES")-$(sha256_file "$CLASSPATH_BYTES")"
GENERATION="$GENERATIONS_DIR/$GENERATION_ID"
if [[ -e "$GENERATION" ]]; then
  [[ -d "$GENERATION" && ! -L "$GENERATION" ]] || fail "unsafe existing generation: $GENERATION"
  rm -rf -- "$STAGE"
else
  mv -- "$STAGE" "$GENERATION"
  chmod -R a-w "$GENERATION"
fi
LINK_TMP="$OUTPUT_LINK.tmp.$$"
ln -s "$GENERATION" "$LINK_TMP"
mv -Tf -- "$LINK_TMP" "$OUTPUT_LINK"
printf 'Published Android CPU importer SDK: %s\n' "$GENERATION"
printf 'Public SDK link: %s\n' "$OUTPUT_LINK"
printf 'Native libraries: %s\n' "$(wc -l <"$GENERATION/metadata/cmake-owned-native-libraries.txt")"
