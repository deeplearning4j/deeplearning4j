#!/usr/bin/env bash
# Build and atomically publish a provenance-bound Android arm64 SDX AOT SDK.
#
# This script intentionally separates the classpath-dependent Graal image and
# JavaCPP bridges from the reusable native CPU/importer closure. Every byte in
# both sets is attested before the public SDK symlink is changed.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: build-android-aot-sdk.sh [options]

Required:
  --android-ndk DIR          Android NDK 28.1.13356709
  --graalvm-home DIR         GraalVM 21.0.10 / Native Image jvmci-23.1-b84
  --object-builder FILE      build-android-ndk.sh with --object-output support
  --maven FILE               Maven executable used for fresh isolated compilation
  --java-home DIR            JDK 17 used for fresh isolated compilation
  --javacpp-jar FILE         JavaCPP 1.5.13 builder jar
  --base-aar FILE            Immutable CPU-importer AAR supplying nd4j-native (legacy)
  --base-sdk DIR             Immutable CPU-importer SDK supplying nd4j-native
  --reuse-jdk-libs DIR       Verified JDK 21.0.10 Android native closure
  --reuse-svm-libs DIR       libjvm/liblibchelper Android archives

Optional:
  --work-dir DIR             Disposable Native Image work root
  --output-link DIR          Public SDK symlink (default: target/android-aot)
  --sdk-version VERSION      Archive identity (default: 1.0.0-SNAPSHOT)
  --offline                  Require object-builder offline mode
  --quick-build              Use Native Image -Ob for rapid development builds
  --production               Use optimized production mode (default)
  --keep-work                Preserve the exact build root for diagnostics
  --fresh-classes-only       Compile and preserve the exact AOT classpath, then stop
  --jobs N                   Native support build jobs
  -h, --help                 Show this help

Native Image cache environment:
  SDX_NATIVE_CACHE=0                     Disable local and shared object reuse
  SDX_NATIVE_FORCE_REBUILD=1             Rebuild despite a valid cache hit
  SDX_NATIVE_CACHE_DIR=/cache/root       Override ~/.cache/sdx/native-images
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODULE_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
DL4J_ROOT="$(cd "$MODULE_DIR/../.." && pwd)"
ANDROID_NDK=""
GRAALVM_HOME=""
OBJECT_BUILDER=""
MAVEN=""
JAVA_HOME_ARG=""
JAVACPP_JAR=""
BASE_SDK=""
BASE_AAR=""
REUSE_JDK_LIBS=""
REUSE_SVM_LIBS=""
WORK_DIR="$MODULE_DIR/target/android-aot-current-build"
OUTPUT_LINK="$MODULE_DIR/target/android-aot"
SDK_VERSION="1.0.0-SNAPSHOT"
JOBS="${JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || printf '8')}"
OFFLINE=0
KEEP_WORK=0
FRESH_CLASSES_ONLY=0
NATIVE_IMAGE_BUILD_MODE=production
NATIVE_IMAGE_OPTIMIZATION=2
NATIVE_IMAGE_OPTIMIZATION_CONFIG_SHA256=none
ANDROID_API=28
EXPECTED_NDK_REVISION=28.1.13356709
EXPECTED_GRAAL_JAVA=21.0.10
EXPECTED_NATIVE_IMAGE=jvmci-23.1-b84
EXPECTED_LABSJDK_URL=https://github.com/graalvm/labs-openjdk-21.git
EXPECTED_LABSJDK_REF=jvmci-23.1-b33
EXPECTED_LABSJDK_COMMIT=ef9d66c6808536e7029680f6f4d965359f8f20c8
EXPECTED_UNIX_FILE_ATTRIBUTES_FIELD=st_birthtime_sec
FORBIDDEN_UNIX_FILE_ATTRIBUTES_FIELD=st_birthtime_nsec

fail() {
  printf 'build-android-aot-sdk: %s\n' "$*" >&2
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

validate_published_native_bytes() {
  local generation="$1"
  local manifest="$generation/metadata/cmake-owned-native-libraries.txt"
  local native_bytes="$generation/metadata/sdk-native-bytes.txt"
  local jni_dir="$generation/jni/arm64-v8a"
  local expected_sha library_name
  [[ -d "$jni_dir" && ! -L "$jni_dir" && -s "$manifest" && -s "$native_bytes" ]] || return 1
  cmp -s \
    <(LC_ALL=C sort -u "$manifest") \
    <(find "$jni_dir" -maxdepth 1 -type f -name '*.so' -printf '%f\n' | LC_ALL=C sort -u) || return 1
  cmp -s \
    <(cut -d ' ' -f 2- "$native_bytes" | LC_ALL=C sort -u) \
    <(LC_ALL=C sort -u "$manifest") || return 1
  while read -r expected_sha library_name; do
    [[ "$expected_sha" =~ ^[0-9a-f]{64}$ && "$library_name" =~ ^lib[A-Za-z0-9._+-]+[.]so$ ]] || return 1
    [[ -f "$jni_dir/$library_name" && ! -L "$jni_dir/$library_name" && -s "$jni_dir/$library_name" ]] || return 1
    [[ "$(sha256_file "$jni_dir/$library_name")" == "$expected_sha" ]] || return 1
  done <"$native_bytes"
}

tree_manifest() {
  local root="$1"
  local file relative mode digest
  [[ -d "$root" ]] || fail "manifest root is missing: $root"
  if find "$root" -type l -print -quit | grep -q .; then
    fail "manifest root contains a symlink: $root"
  fi
  while IFS= read -r -d '' file; do
    relative="${file#"$root"/}"
    mode="$(stat -c '%a' "$file")"
    digest="$(sha256_file "$file")"
    printf '%s\0%s\0%s\0' "$relative" "$mode" "$digest"
  done < <(find "$root" -type f -print0 | LC_ALL=C sort -z)
}

tree_manifest_sha256() {
  tree_manifest "$1" | sha256sum | cut -d ' ' -f 1
}

assert_unix_file_attributes_abi() {
  local binary="$1"
  [[ -s "$binary" ]] || fail "UnixFileAttributes ABI verification input is missing: $binary"
  LC_ALL=C grep -a -F -q "$EXPECTED_UNIX_FILE_ATTRIBUTES_FIELD" "$binary" ||
    fail "$binary does not reference required field $EXPECTED_UNIX_FILE_ATTRIBUTES_FIELD"
  if LC_ALL=C grep -a -F -q "$FORBIDDEN_UNIX_FILE_ATTRIBUTES_FIELD" "$binary"; then
    fail "$binary references forbidden GraalVM field $FORBIDDEN_UNIX_FILE_ATTRIBUTES_FIELD"
  fi
}

SOURCE_MANIFEST_HELPER="$SCRIPT_DIR/source-manifest.sh"
[[ -r "$SOURCE_MANIFEST_HELPER" ]] ||
  fail "source manifest helper is missing: $SOURCE_MANIFEST_HELPER"
# shellcheck source=source-manifest.sh
source "$SOURCE_MANIFEST_HELPER"
MODULE_RESOURCE_STAGER="$SCRIPT_DIR/stage-module-resources.sh"
[[ -r "$MODULE_RESOURCE_STAGER" ]] ||
  fail "module resource stager is missing: $MODULE_RESOURCE_STAGER"
# shellcheck source=stage-module-resources.sh
source "$MODULE_RESOURCE_STAGER"
NATIVE_IMAGE_CACHE_HELPER="$SCRIPT_DIR/native-image-cache.sh"
[[ -r "$NATIVE_IMAGE_CACHE_HELPER" ]] ||
  fail "Native Image cache helper is missing: $NATIVE_IMAGE_CACHE_HELPER"
# shellcheck source=native-image-cache.sh
source "$NATIVE_IMAGE_CACHE_HELPER"
sdx_native_cache_configure || fail "invalid Native Image cache configuration"
NATIVE_IMAGE_OBJECT_IDENTITY_HELPER="$SCRIPT_DIR/native-image-object-identity.sh"
[[ -r "$NATIVE_IMAGE_OBJECT_IDENTITY_HELPER" ]] ||
  fail "Native Image object identity helper is missing: $NATIVE_IMAGE_OBJECT_IDENTITY_HELPER"
NATIVE_IMAGE_OBJECT_IDENTITY_HELPER_SHA256="$(sha256_file "$NATIVE_IMAGE_OBJECT_IDENTITY_HELPER")"
# shellcheck source=native-image-object-identity.sh
source "$NATIVE_IMAGE_OBJECT_IDENTITY_HELPER"
# Keep orchestration-only wrappers outside the AOT source identity. The base SDK
# receipt already binds libnd4j's native bytes, while the roots below are the
# managed sources and build metadata that can actually change this image.
DL4J_AOT_SOURCE_ROOTS=(
  pom.xml
  build-scripts/release/native-platform.sh
  nd4j/pom.xml
  nd4j/nd4j-backends/pom.xml
  nd4j/nd4j-backends/nd4j-api-parent/pom.xml
  nd4j/nd4j-backends/nd4j-backend-impls/pom.xml
  nd4j/sdx-aot/pom.xml
  nd4j/sdx-aot/include
  nd4j/sdx-aot/src/main/java
  nd4j/sdx-aot/src/main/resources
  nd4j/sdx-aot/src/main/assembly
  nd4j/sdx-aot/src/main/linker
  nd4j/sdx-aot/src/main/android/EmbeddedClasspathAudit.java
  nd4j/sdx-aot/src/main/android/JavaCppNativeImageReachability.java
  nd4j/sdx-aot/src/main/android/source-manifest.sh
  nd4j/sdx-aot/src/main/android/stage-module-resources.sh
  nd4j/nd4j-backends/nd4j-api-parent/nd4j-api
  nd4j/nd4j-backends/nd4j-api-parent/nd4j-native-api
  nd4j/nd4j-backends/nd4j-backend-impls/nd4j-presets-common
  nd4j/nd4j-backends/nd4j-backend-impls/nd4j-native-preset
  nd4j/nd4j-backends/nd4j-backend-impls/nd4j-cpu-backend-common
  nd4j/nd4j-backends/nd4j-backend-impls/nd4j-native
  nd4j/nd4j-tokenizers/tokenizers-native-preset
  nd4j/nd4j-tokenizers/tokenizers-native
  nd4j/nd4j-ggml
  nd4j/samediff-llm
  nd4j/nd4j-backends/nd4j-backend-impls/nd4j-sdx
  nd4j/nd4j-backends/nd4j-backend-impls/nd4j-sdx-model
  nd4j/nd4j-backends/nd4j-backend-impls/nd4j-sdx-preset
)

module_source_manifest_sha256() {
  local root="${1:?module source root is required}"
  sdx_git_source_manifest_sha256 "$DL4J_ROOT" "$root"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --android-ndk) ANDROID_NDK="${2:?missing value for --android-ndk}"; shift 2 ;;
    --graalvm-home) GRAALVM_HOME="${2:?missing value for --graalvm-home}"; shift 2 ;;
    --object-builder) OBJECT_BUILDER="${2:?missing value for --object-builder}"; shift 2 ;;
    --maven) MAVEN="${2:?missing value for --maven}"; shift 2 ;;
    --java-home) JAVA_HOME_ARG="${2:?missing value for --java-home}"; shift 2 ;;
    --javacpp-jar) JAVACPP_JAR="${2:?missing value for --javacpp-jar}"; shift 2 ;;
    --base-aar) BASE_AAR="${2:?missing value for --base-aar}"; shift 2 ;;
    --base-sdk) BASE_SDK="${2:?missing value for --base-sdk}"; shift 2 ;;
    --reuse-jdk-libs) REUSE_JDK_LIBS="${2:?missing value for --reuse-jdk-libs}"; shift 2 ;;
    --reuse-svm-libs) REUSE_SVM_LIBS="${2:?missing value for --reuse-svm-libs}"; shift 2 ;;
    --work-dir) WORK_DIR="${2:?missing value for --work-dir}"; shift 2 ;;
    --output-link) OUTPUT_LINK="${2:?missing value for --output-link}"; shift 2 ;;
    --sdk-version) SDK_VERSION="${2:?missing value for --sdk-version}"; shift 2 ;;
    --jobs) JOBS="${2:?missing value for --jobs}"; shift 2 ;;
    --offline) OFFLINE=1; shift ;;
    --quick-build) NATIVE_IMAGE_BUILD_MODE=dev; NATIVE_IMAGE_OPTIMIZATION=b; shift ;;
    --production) NATIVE_IMAGE_BUILD_MODE=production; NATIVE_IMAGE_OPTIMIZATION=2; shift ;;
    --keep-work) KEEP_WORK=1; shift ;;
    --fresh-classes-only) FRESH_CLASSES_ONLY=1; KEEP_WORK=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) fail "unknown argument: $1" ;;
  esac
done

for required in ANDROID_NDK GRAALVM_HOME OBJECT_BUILDER MAVEN JAVA_HOME_ARG JAVACPP_JAR REUSE_JDK_LIBS REUSE_SVM_LIBS; do
  [[ -n "${!required}" ]] || fail "required argument is missing: $required"
done
[[ -n "$BASE_AAR" || -n "$BASE_SDK" ]] || fail "a CPU importer closure is required via --base-aar or --base-sdk"
[[ -z "$BASE_AAR" || -z "$BASE_SDK" ]] || fail "--base-aar and --base-sdk are mutually exclusive"
[[ "$SDK_VERSION" =~ ^[A-Za-z0-9._-]+$ ]] || fail "unsafe SDK version: $SDK_VERSION"
[[ "$JOBS" =~ ^[1-9][0-9]*$ ]] || fail "jobs must be a positive integer"
[[ "$NATIVE_IMAGE_BUILD_MODE" == dev || "$NATIVE_IMAGE_BUILD_MODE" == production ]] ||
  fail "Native Image build mode must be dev or production"
if [[ "$NATIVE_IMAGE_BUILD_MODE" == dev ]]; then
  NATIVE_IMAGE_OPTIMIZATION_CONFIG_SHA256="$(printf '%s\n' 'Args = -Ob' | sha256sum | cut -d ' ' -f 1)"
fi
[[ -x "$OBJECT_BUILDER" ]] || fail "object builder is not executable: $OBJECT_BUILDER"
[[ -x "$MAVEN" ]] || fail "Maven executable is missing: $MAVEN"
[[ -x "$JAVA_HOME_ARG/bin/java" && -x "$JAVA_HOME_ARG/bin/javac" ]] ||
  fail "JDK installation is incomplete: $JAVA_HOME_ARG"
MAVEN="$(realpath -e -- "$MAVEN")"
JAVA_HOME_ARG="$(realpath -e -- "$JAVA_HOME_ARG")"
JAVA_SPECIFICATION_VERSION="$("$JAVA_HOME_ARG/bin/java" -XshowSettings:properties -version 2>&1 | sed -n 's/^[[:space:]]*java[.]specification[.]version = //p')"
[[ "$JAVA_SPECIFICATION_VERSION" == 17 ]] || fail "fresh compilation requires JDK 17"
[[ -s "$JAVACPP_JAR" ]] || fail "JavaCPP builder jar is missing: $JAVACPP_JAR"
if [[ -n "$BASE_AAR" ]]; then
  [[ -f "$BASE_AAR" && ! -L "$BASE_AAR" && -s "$BASE_AAR" ]] ||
    fail "base AAR is missing, empty, or not a regular file: $BASE_AAR"
  command -v unzip >/dev/null || fail "unzip is required for --base-aar"
else
  [[ -d "$BASE_SDK/jni/arm64-v8a" ]] || fail "base SDK JNI directory is missing"
  [[ -s "$BASE_SDK/metadata/cmake-owned-native-libraries.txt" ]] ||
    fail "base SDK native manifest is missing"
fi
[[ -x "$GRAALVM_HOME/bin/native-image" && -x "$GRAALVM_HOME/bin/java" ]] ||
  fail "GraalVM installation is incomplete: $GRAALVM_HOME"
[[ -s "$ANDROID_NDK/source.properties" ]] || fail "NDK source.properties is missing"

NDK_REVISION="$(awk -F= '/Pkg.Revision/ {gsub(/[[:space:]]/, "", $2); print $2}' "$ANDROID_NDK/source.properties")"
[[ "$NDK_REVISION" == "$EXPECTED_NDK_REVISION" ]] ||
  fail "expected NDK $EXPECTED_NDK_REVISION, found $NDK_REVISION"
NATIVE_IMAGE_VERSION="$("$GRAALVM_HOME/bin/native-image" --version 2>&1)"
[[ "$NATIVE_IMAGE_VERSION" == *"$EXPECTED_GRAAL_JAVA"* &&
   "$NATIVE_IMAGE_VERSION" == *"$EXPECTED_NATIVE_IMAGE"* ]] ||
  fail "unexpected GraalVM Native Image: $NATIVE_IMAGE_VERSION"

HOST_TAG="${NDK_HOST_TAG:-linux-x86_64}"
TOOLCHAIN="$ANDROID_NDK/toolchains/llvm/prebuilt/$HOST_TAG/bin"
CLANG="$TOOLCHAIN/aarch64-linux-android${ANDROID_API}-clang"
CLANGXX="$TOOLCHAIN/aarch64-linux-android${ANDROID_API}-clang++"
LLVM_NM="$TOOLCHAIN/llvm-nm"
LLVM_READELF="$TOOLCHAIN/llvm-readelf"
LLVM_STRIP="$TOOLCHAIN/llvm-strip"
for tool in "$CLANG" "$CLANGXX" "$LLVM_NM" "$LLVM_READELF" "$LLVM_STRIP"; do
  [[ -x "$tool" ]] || fail "required NDK tool is missing: $tool"
done
CXX_SHARED="$ANDROID_NDK/toolchains/llvm/prebuilt/$HOST_TAG/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so"
[[ -s "$CXX_SHARED" ]] || fail "NDK libc++_shared.so is missing"

REUSE_JDK_LIBS_SOURCE="$(realpath -e -- "$REUSE_JDK_LIBS")"
REUSE_SVM_LIBS_SOURCE="$(realpath -e -- "$REUSE_SVM_LIBS")"
for archive in libjava.a libnet.a libnio.a libzip.a libprefs.a libextnet.a; do
  [[ -f "$REUSE_JDK_LIBS_SOURCE/$archive" && ! -L "$REUSE_JDK_LIBS_SOURCE/$archive" &&
     -s "$REUSE_JDK_LIBS_SOURCE/$archive" ]] || fail "JDK support archive is missing or unsafe: $archive"
done
for archive in libjvm.a liblibchelper.a; do
  [[ -f "$REUSE_SVM_LIBS_SOURCE/$archive" && ! -L "$REUSE_SVM_LIBS_SOURCE/$archive" &&
     -s "$REUSE_SVM_LIBS_SOURCE/$archive" ]] || fail "SVM support archive is missing or unsafe: $archive"
done

OBJECT_BUILDER="$(realpath -e -- "$OBJECT_BUILDER")"
OBJECT_BUILDER_SHA256="$(sha256_file "$OBJECT_BUILDER")"
JDK_SUPPORT_RECEIPT="$REUSE_JDK_LIBS_SOURCE/jdk-support-receipt"
[[ -s "$JDK_SUPPORT_RECEIPT" ]] ||
  fail "JDK support closure receipt is missing: $JDK_SUPPORT_RECEIPT"
JDK_SUPPORT_RECEIPT_SHA256="$(sha256_file "$JDK_SUPPORT_RECEIPT")"
declare -A JDK_ARCHIVE_SHA256=()
for expected in \
  "format=1" \
  "stage=android-jdk-support" \
  "java_version=$EXPECTED_GRAAL_JAVA" \
  "labsjdk_source_url=$EXPECTED_LABSJDK_URL" \
  "labsjdk_source_ref=$EXPECTED_LABSJDK_REF" \
  "labsjdk_source_commit=$EXPECTED_LABSJDK_COMMIT" \
  "android_ndk_revision=$NDK_REVISION" \
  "android_ndk_source_properties_sha256=$(sha256_file "$ANDROID_NDK/source.properties")" \
  "graalvm_version_sha256=$(printf '%s' "$NATIVE_IMAGE_VERSION" | sha256sum | cut -d ' ' -f 1)"; do
  [[ "$(grep -Fxc "$expected" "$JDK_SUPPORT_RECEIPT")" -eq 1 ]] ||
    fail "JDK support receipt is missing or duplicates: $expected"
done
# The receipt's producer hash is provenance, not the compatibility key for the
# JDK archive closure. The object builder also owns Native Image/relink logic;
# unrelated edits there must not invalidate verified, pinned JDK archives.
[[ "$(grep -Ec '^producer_sha256=[0-9a-f]{64}$' "$JDK_SUPPORT_RECEIPT")" -eq 1 ]] ||
  fail "JDK support receipt has invalid producer provenance"
for archive in libjava.a libnet.a libnio.a libzip.a libprefs.a libextnet.a; do
  JDK_ARCHIVE_SHA256["$archive"]="$(sha256_file "$REUSE_JDK_LIBS_SOURCE/$archive")"
  expected="${archive%.a}_sha256=${JDK_ARCHIVE_SHA256[$archive]}"
  [[ "$(grep -Fxc "$expected" "$JDK_SUPPORT_RECEIPT")" -eq 1 ]] ||
    fail "JDK support receipt does not bind $archive"
done
[[ "$(sha256_file "$JDK_SUPPORT_RECEIPT")" == "$JDK_SUPPORT_RECEIPT_SHA256" ]] ||
  fail "JDK support receipt changed during validation"
if find "$REUSE_JDK_LIBS_SOURCE" -maxdepth 1 -type l -print -quit | grep -q .; then
  fail "JDK support closure contains a symlink"
fi

WORK_DIR="$(realpath -m -- "$WORK_DIR")"
mkdir -p "$WORK_DIR"
require_static_symbol() {
  local archive="$1"
  local symbol="$2"
  local symbols_file="$WORK_DIR/$(basename "$archive").required-symbols"
  "$LLVM_NM" --defined-only "$archive" >"$symbols_file"
  grep -q "[[:space:]]$symbol$" "$symbols_file" ||
    fail "JDK support archive $(basename "$archive") omits $symbol"
}
require_static_symbol "$REUSE_JDK_LIBS_SOURCE/libnio.a" Java_sun_nio_ch_Net_shouldShutdownWriteBeforeClose0
require_static_symbol "$REUSE_JDK_LIBS_SOURCE/libprefs.a" Java_java_util_prefs_FileSystemPreferences_lockFile0
require_static_symbol "$REUSE_JDK_LIBS_SOURCE/libextnet.a" Java_jdk_net_LinuxSocketOptions_setQuickAck0
LIBJVM_SHA256="$(sha256_file "$REUSE_SVM_LIBS_SOURCE/libjvm.a")"
LIBLIBCHELPER_SHA256="$(sha256_file "$REUSE_SVM_LIBS_SOURCE/liblibchelper.a")"

OUTPUT_LINK_BASENAME="$(basename -- "$OUTPUT_LINK")"
OUTPUT_PARENT="$(realpath -m -- "$(dirname -- "$OUTPUT_LINK")")"
OUTPUT_LINK="$OUTPUT_PARENT/$OUTPUT_LINK_BASENAME"
GENERATIONS_DIR="$OUTPUT_PARENT/.android-aot-generations"
OBJECT_STAGES_DIR="$WORK_DIR/native-image-object-stages"
mkdir -p "$WORK_DIR" "$GENERATIONS_DIR" "$OBJECT_STAGES_DIR"
BUILD_ROOT="$(mktemp -d "$WORK_DIR/generation.XXXXXXXX")"
cleanup_build_root() {
  local status=$?
  trap - EXIT
  if [[ "$KEEP_WORK" == 1 ]]; then
    printf 'Preserved SDX Android AOT build root: %s\n' "$BUILD_ROOT" >&2
  else
    chmod -R u+w "$BUILD_ROOT" 2>/dev/null || true
    rm -rf -- "$BUILD_ROOT"
  fi
  return "$status"
}
trap cleanup_build_root EXIT
OBJECT="$BUILD_ROOT/libsdx_llm.o"
OBJECT_WORK="$BUILD_ROOT/native-image"
BRIDGE_DIR="$BUILD_ROOT/bridge"
STAGE="$BUILD_ROOT/sdk"
JNI_DIR="$STAGE/jni/arm64-v8a"
METADATA_DIR="$STAGE/metadata"
INCLUDE_DIR="$STAGE/include"
mkdir -p "$BRIDGE_DIR" "$JNI_DIR" "$METADATA_DIR" "$INCLUDE_DIR"

copy_static_archive() {
  local source="$1"
  local target="$2"
  local expected_sha256="$3"
  cp --reflink=auto -- "$source" "$target"
  [[ "$(stat -c '%d:%i' "$source")" != "$(stat -c '%d:%i' "$target")" ]] ||
    fail "static input was hard-linked into the build: $source"
  [[ "$(sha256_file "$target")" == "$expected_sha256" &&
     "$(sha256_file "$source")" == "$expected_sha256" ]] ||
    fail "static input changed during independent staging: $source"
  chmod a-w "$target"
}

LOCAL_JDK_LIBS="$BUILD_ROOT/static-inputs/jdk"
LOCAL_SVM_LIBS="$BUILD_ROOT/static-inputs/svm"
mkdir -p "$LOCAL_JDK_LIBS" "$LOCAL_SVM_LIBS"
for archive in libjava.a libnet.a libnio.a libzip.a libprefs.a libextnet.a; do
  copy_static_archive \
    "$REUSE_JDK_LIBS_SOURCE/$archive" \
    "$LOCAL_JDK_LIBS/$archive" \
    "${JDK_ARCHIVE_SHA256[$archive]}"
done
copy_static_archive "$JDK_SUPPORT_RECEIPT" "$LOCAL_JDK_LIBS/jdk-support-receipt" "$JDK_SUPPORT_RECEIPT_SHA256"
copy_static_archive "$REUSE_SVM_LIBS_SOURCE/libjvm.a" "$LOCAL_SVM_LIBS/libjvm.a" "$LIBJVM_SHA256"
copy_static_archive "$REUSE_SVM_LIBS_SOURCE/liblibchelper.a" "$LOCAL_SVM_LIBS/liblibchelper.a" "$LIBLIBCHELPER_SHA256"
REUSE_JDK_LIBS="$LOCAL_JDK_LIBS"
REUSE_SVM_LIBS="$LOCAL_SVM_LIBS"
JDK_SUPPORT_RECEIPT="$LOCAL_JDK_LIBS/jdk-support-receipt"
SOURCE_MANIFEST_SHA256="$(sdx_git_source_manifest_sha256 "$DL4J_ROOT" "${DL4J_AOT_SOURCE_ROOTS[@]}")"
BASE_SDK_RECEIPT_SHA256=not-applicable

if [[ -n "$BASE_AAR" ]]; then
  BASE_SDK_INPUT="$(realpath -e -- "$BASE_AAR")"
  BASE_SDK_INPUT_SHA256="$(sha256_file "$BASE_SDK_INPUT")"
  BASE_AAR_NAMES="$BUILD_ROOT/base-aar.names"
  unzip -Z1 "$BASE_SDK_INPUT" >"$BASE_AAR_NAMES" || fail "base AAR is not a readable ZIP"
  [[ "$(wc -l <"$BASE_AAR_NAMES")" -eq "$(LC_ALL=C sort -u "$BASE_AAR_NAMES" | wc -l)" ]] ||
    fail "base AAR has duplicate members"
  while IFS= read -r member; do
    [[ -n "$member" && "$member" != /* && "$member" != ../* &&
       "$member" != */../* && "$member" != */.. ]] ||
      fail "base AAR has an unsafe member: $member"
  done <"$BASE_AAR_NAMES"
  if grep -E '^jni/.+/lib[^/]+[.]so$' "$BASE_AAR_NAMES" |
       grep -Evq '^jni/arm64-v8a/lib[A-Za-z0-9._+-]+[.]so$'; then
    fail "base AAR contains a native library outside arm64-v8a"
  fi
  mapfile -t BASE_AAR_NATIVE_MEMBERS < <(
    grep -E '^jni/arm64-v8a/lib[A-Za-z0-9._+-]+[.]so$' "$BASE_AAR_NAMES" | LC_ALL=C sort -u
  )
  [[ ${#BASE_AAR_NATIVE_MEMBERS[@]} -gt 0 ]] || fail "base AAR has no arm64 native libraries"
  BASE_SDK="$BUILD_ROOT/base-sdk"
  mkdir -p "$BASE_SDK/metadata"
  unzip -q "$BASE_SDK_INPUT" 'jni/arm64-v8a/*.so' -d "$BASE_SDK"
  if find "$BASE_SDK" -type l -print -quit | grep -q .; then
    fail "base AAR extracted a symlink"
  fi
  find "$BASE_SDK/jni/arm64-v8a" -maxdepth 1 -type f -name '*.so' -printf '%f\n' |
    LC_ALL=C sort -u >"$BASE_SDK/metadata/cmake-owned-native-libraries.txt"
  [[ "$(wc -l <"$BASE_SDK/metadata/cmake-owned-native-libraries.txt")" -eq "${#BASE_AAR_NATIVE_MEMBERS[@]}" ]] ||
    fail "base AAR native extraction disagrees with its member list"
  chmod -R a-w "$BASE_SDK"
else
  BASE_SDK="$(realpath -e -- "$BASE_SDK")"
  BASE_SDK_INPUT="$BASE_SDK"
  if find "$BASE_SDK" -type l -print -quit | grep -q .; then
    fail "base SDK generation contains a symlink: $BASE_SDK"
  fi
  if find "$BASE_SDK" -perm /0222 -print -quit | grep -q .; then
    fail "base SDK generation contains a writable member: $BASE_SDK"
  fi
  BASE_SDK_INPUT_SHA256="$(tree_manifest_sha256 "$BASE_SDK_INPUT")"
  BASE_SDK_RECEIPT="$BASE_SDK/metadata/build-receipt"
  [[ -f "$BASE_SDK_RECEIPT" && ! -L "$BASE_SDK_RECEIPT" && -s "$BASE_SDK_RECEIPT" ]] ||
    fail "base SDK build receipt is missing, empty, or unsafe"
  BASE_SDK_RECEIPT_SHA256="$(sha256_file "$BASE_SDK_RECEIPT")"
  CPU_IMPORTER_PRODUCER="$(realpath -e -- "$SCRIPT_DIR/build-android-cpu-importer-sdk.sh")"
  for expected in \
    "format=2" \
    "stage=android-cpu-importer-sdk" \
    "cache_schema=independent-stages-v1" \
    "producer=$CPU_IMPORTER_PRODUCER" \
    "producer_sha256=$(sha256_file "$CPU_IMPORTER_PRODUCER")" \
    "android_api=$ANDROID_API" \
    "android_abi=arm64-v8a" \
    "process_blas_symbols_abi=1" \
    "process_blas_symbols_capability=nd4j_process_blas_symbols_abi_v1" \
    "classpath_bytes_sha256=$(sha256_file "$BASE_SDK/metadata/classpath-bytes.txt")" \
    "native_bytes_sha256=$(sha256_file "$BASE_SDK/metadata/native-bytes.txt")"; do
    [[ "$(grep -Fxc "$expected" "$BASE_SDK_RECEIPT")" -eq 1 ]] ||
      fail "base SDK receipt is missing or duplicates: $expected"
  done
  [[ "$(sha256_file "$BASE_SDK_RECEIPT")" == "$BASE_SDK_RECEIPT_SHA256" ]] ||
    fail "base SDK build receipt changed during validation"
fi

# The AOT image owns raw GGUF conversion. The accelerator provider is selected
# independently by the Android flavor, so a provider-only AAR is never a valid
# base for this SDK. Require the exact JavaCPP CPU backend and its native math
# closure before doing any expensive Native Image work.
for importer_library in libjnind4jcpu.so libnd4jcpu.so libopenblas.so libomp.so; do
  grep -Fxq "$importer_library" "$BASE_SDK/metadata/cmake-owned-native-libraries.txt" ||
    fail "CPU importer closure is missing required library: $importer_library"
  [[ -f "$BASE_SDK/jni/arm64-v8a/$importer_library" &&
     ! -L "$BASE_SDK/jni/arm64-v8a/$importer_library" &&
     -s "$BASE_SDK/jni/arm64-v8a/$importer_library" ]] ||
    fail "CPU importer library is missing, empty, or unsafe: $importer_library"
done
for provider_library in libnd4jnnapi.so libnd4jvulkan.so liblitert-lm.so; do
  if grep -Fxq "$provider_library" "$BASE_SDK/metadata/cmake-owned-native-libraries.txt"; then
    fail "CPU importer closure contains accelerator provider library: $provider_library"
  fi
done
CPU_BACKEND="$BASE_SDK/jni/arm64-v8a/libnd4jcpu.so"
PROCESS_BLAS_SYMBOLS_ABI=nd4j_process_blas_symbols_abi_v1
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

# Resolve every inexpensive external identity before invoking Maven or Native Image.
# A completed generation is immutable and content-addressed, so an exact receipt match
# can return immediately instead of rebuilding hours of already-proven output.
LINKER_SCRIPT="$MODULE_DIR/src/main/linker/sdx_exports.lds"
[[ -s "$LINKER_SCRIPT" ]] || fail "linker version script is missing: $LINKER_SCRIPT"
BUILD_SCRIPT_SHA256="$(sha256_file "${BASH_SOURCE[0]}")"
NATIVE_IMAGE_CACHE_HELPER_SHA256="$(sha256_file "$NATIVE_IMAGE_CACHE_HELPER")"
MAVEN_SHA256="$(sha256_file "$MAVEN")"
MAVEN_VERSION="$({ env -u JAVA_TOOL_OPTIONS JAVA_HOME="$JAVA_HOME_ARG" PATH="$JAVA_HOME_ARG/bin:$PATH" "$MAVEN" --version; } 2>&1)"
MAVEN_VERSION_SHA256="$(printf '%s\n' "$MAVEN_VERSION" | sha256sum | cut -d ' ' -f 1)"
JAVA_VERSION="$({ env -u JAVA_TOOL_OPTIONS "$JAVA_HOME_ARG/bin/java" -version; } 2>&1)"
JAVA_VERSION_SHA256="$(printf '%s\n' "$JAVA_VERSION" | sha256sum | cut -d ' ' -f 1)"
LINKER_SCRIPT_SHA256="$(sha256_file "$LINKER_SCRIPT")"
JAVACPP_JAR_SHA256="$(sha256_file "$JAVACPP_JAR")"
NDK_REVISION_SHA256="$(sha256_file "$ANDROID_NDK/source.properties")"
GRAALVM_VERSION_SHA256="$(printf '%s' "$NATIVE_IMAGE_VERSION" | sha256sum | cut -d ' ' -f 1)"

validate_published_generation() {
  local generation receipt native_bytes complete receipt_sha archive_name archive archive_sha
  [[ "$FRESH_CLASSES_ONLY" == 0 && -L "$OUTPUT_LINK" ]] || return 1
  generation="$(realpath -e -- "$OUTPUT_LINK")" || return 1
  case "$generation/" in
    "$GENERATIONS_DIR"/*/) ;;
    *) return 1 ;;
  esac
  [[ -d "$generation" && ! -L "$generation" ]] || return 1
  ! find "$generation" -type l -print -quit | grep -q . || return 1
  ! find "$generation" -perm /0222 -print -quit | grep -q . || return 1
  receipt="$generation/metadata/build-receipt"
  native_bytes="$generation/metadata/sdk-native-bytes.txt"
  complete="$generation/.complete.cmake"
  [[ -s "$receipt" && -s "$native_bytes" && -s "$complete" ]] || return 1
  for expected in \
    "format=9" \
    "stage=android-aot-sdk" \
    "native_image_build_mode=$NATIVE_IMAGE_BUILD_MODE" \
    "native_image_optimization=$NATIVE_IMAGE_OPTIMIZATION" \
    "native_image_optimization_config_sha256=$NATIVE_IMAGE_OPTIMIZATION_CONFIG_SHA256" \
    "source_manifest_sha256=$SOURCE_MANIFEST_SHA256" \
    "base_sdk_sha256=$BASE_SDK_INPUT_SHA256" \
    "base_sdk_receipt_sha256=$BASE_SDK_RECEIPT_SHA256" \
    "process_blas_symbols_abi=1" \
    "process_blas_symbols_capability=$PROCESS_BLAS_SYMBOLS_ABI" \
    "build_script_sha256=$BUILD_SCRIPT_SHA256" \
    "native_image_cache_helper_sha256=$NATIVE_IMAGE_CACHE_HELPER_SHA256" \
    "object_builder_sha256=$OBJECT_BUILDER_SHA256" \
    "jdk_support_receipt_sha256=$JDK_SUPPORT_RECEIPT_SHA256" \
    "libjvm_sha256=$LIBJVM_SHA256" \
    "liblibchelper_sha256=$LIBLIBCHELPER_SHA256" \
    "maven_sha256=$MAVEN_SHA256" \
    "maven_version_sha256=$MAVEN_VERSION_SHA256" \
    "java_version_sha256=$JAVA_VERSION_SHA256" \
    "linker_script_sha256=$LINKER_SCRIPT_SHA256" \
    "javacpp_jar_sha256=$JAVACPP_JAR_SHA256" \
    "ndk_revision_sha256=$NDK_REVISION_SHA256" \
    "graalvm_version_sha256=$GRAALVM_VERSION_SHA256" \
    "android_api=$ANDROID_API" \
    "android_abi=arm64-v8a" \
    "sdk_native_bytes_sha256=$(sha256_file "$native_bytes")"; do
    receipt_has "$receipt" "$expected" || return 1
  done
  validate_published_native_bytes "$generation" || return 1
  receipt_sha="$(sha256_file "$receipt")"
  archive_name="sdx-aot-$SDK_VERSION-android-arm64.zip"
  archive="$generation/$archive_name"
  [[ -f "$archive" && ! -L "$archive" && -s "$archive" ]] || return 1
  archive_sha="$(sha256_file "$archive")"
  grep -Fqx -- "set(SDX_COMPLETED_GENERATION_KEY [[$receipt_sha]])" "$complete" || return 1
  grep -Fqx -- "set(SDX_COMPLETED_ARCHIVE_NAME [[$archive_name]])" "$complete" || return 1
  grep -Fqx -- "set(SDX_COMPLETED_ARCHIVE_SHA256 [[$archive_sha]])" "$complete" || return 1
  printf 'Reusing validated SDX Android AOT SDK: %s\n' "$generation"
}

if validate_published_generation; then
  exit 0
fi
printf 'SDX Android AOT cache miss; building a new content-addressed generation.\n'

FRESH_CLASSES_ROOT="$BUILD_ROOT/fresh-classes"
NATIVE_RUNTIME_CLASSES="$FRESH_CLASSES_ROOT/nd4j-native-runtime"
mkdir -p "$FRESH_CLASSES_ROOT" "$NATIVE_RUNTIME_CLASSES"
declare -A MODULE_ROOTS=(
  [nd4j-api]="nd4j/nd4j-backends/nd4j-api-parent/nd4j-api"
  [nd4j-native-api]="nd4j/nd4j-backends/nd4j-api-parent/nd4j-native-api"
  [nd4j-presets-common]="nd4j/nd4j-backends/nd4j-backend-impls/nd4j-presets-common"
  [nd4j-native-preset]="nd4j/nd4j-backends/nd4j-backend-impls/nd4j-native-preset"
  [nd4j-cpu-backend-common]="nd4j/nd4j-backends/nd4j-backend-impls/nd4j-cpu-backend-common"
  [nd4j-native]="nd4j/nd4j-backends/nd4j-backend-impls/nd4j-native"
  [tokenizers-native-preset]="nd4j/nd4j-tokenizers/tokenizers-native-preset"
  [tokenizers-native]="nd4j/nd4j-tokenizers/tokenizers-native"
  [nd4j-ggml]="nd4j/nd4j-ggml"
  [samediff-llm]="nd4j/samediff-llm"
  [nd4j-sdx-preset]="nd4j/nd4j-backends/nd4j-backend-impls/nd4j-sdx-preset"
  [nd4j-sdx-model]="nd4j/nd4j-backends/nd4j-backend-impls/nd4j-sdx-model"
  [nd4j-sdx]="nd4j/nd4j-backends/nd4j-backend-impls/nd4j-sdx"
  [sdx-aot]="nd4j/sdx-aot"
)
declare -A MODULE_CLASS_DIRS=(
  [nd4j-api]="$NATIVE_RUNTIME_CLASSES"
  [nd4j-native-api]="$NATIVE_RUNTIME_CLASSES"
  [nd4j-presets-common]="$NATIVE_RUNTIME_CLASSES"
  [nd4j-native-preset]="$NATIVE_RUNTIME_CLASSES"
  [nd4j-cpu-backend-common]="$NATIVE_RUNTIME_CLASSES"
  [nd4j-native]="$NATIVE_RUNTIME_CLASSES"
  [tokenizers-native-preset]="$FRESH_CLASSES_ROOT/tokenizers-native-preset"
  [tokenizers-native]="$FRESH_CLASSES_ROOT/tokenizers-native"
  [nd4j-ggml]="$FRESH_CLASSES_ROOT/nd4j-ggml"
  [samediff-llm]="$FRESH_CLASSES_ROOT/samediff-llm"
  [nd4j-sdx-preset]="$FRESH_CLASSES_ROOT/nd4j-sdx-preset"
  [nd4j-sdx-model]="$FRESH_CLASSES_ROOT/nd4j-sdx-model"
  [nd4j-sdx]="$FRESH_CLASSES_ROOT/nd4j-sdx"
  [sdx-aot]="$FRESH_CLASSES_ROOT/sdx-aot"
)
declare -A MODULE_PROBE_CLASSES=(
  [nd4j-api]="org/nd4j/nativeblas/NativeSymbolResolution.class"
  [nd4j-native-api]="org/nd4j/nativeblas/BaseNativeNDArrayFactory.class"
  [nd4j-presets-common]="org/nd4j/presets/OpExclusionUtils.class"
  [nd4j-native-preset]="org/nd4j/presets/cpu/Nd4jCpuHelper.class"
  [nd4j-cpu-backend-common]="org/nd4j/linalg/cpu/nativecpu/CpuNDArrayFactory.class"
  [nd4j-native]="org/nd4j/linalg/cpu/nativecpu/CpuBackend.class"
  [tokenizers-native-preset]="org/eclipse/deeplearning4j/tokenizers/presets/TokenizersPresets.class"
  [tokenizers-native]="org/eclipse/deeplearning4j/tokenizers/NativeTokenizer.class"
  [nd4j-ggml]="org/nd4j/ggml/GGMLModelImport.class"
  [samediff-llm]="org/eclipse/deeplearning4j/llm/tokenizer/HuggingFaceTokenizer.class"
  [nd4j-sdx-preset]="org/nd4j/dsp/runtime/presets/SdxRuntimePresets.class"
  [nd4j-sdx-model]="org/nd4j/dsp/model/SdxLlmNative.class"
  [nd4j-sdx]="org/nd4j/dsp/runtime/SdxRuntime.class"
  [sdx-aot]="org/eclipse/deeplearning4j/sdx/aot/SdxLlmCore.class"
)
FRESH_COMPILE_ORDER=(
  nd4j-api
  nd4j-native-api
  nd4j-presets-common
  nd4j-native-preset
  nd4j-cpu-backend-common
  nd4j-native
  tokenizers-native-preset
  tokenizers-native
  nd4j-ggml
  samediff-llm
  nd4j-sdx-preset
  nd4j-sdx-model
  nd4j-sdx
  sdx-aot
)
# This reactor runs on the Linux build host but produces the managed closure for
# an Android/OpenBLAS image. Bind both decisions explicitly so host-only JavaCPP
# profiles (notably MKL) cannot leak into the cross-target classpath.
maven_compile_flags=(
  -DskipTests
  -Dlibnd4j.blas=openblas
  -Djavacpp.platform=android-arm64
)
[[ "$OFFLINE" == 0 ]] || maven_compile_flags+=(-o)
MAVEN_DEPENDENCY_ARGUMENTS_SHA256="$(
  printf '%s\n' \
    "profile=cpu-managed" \
    "reactor_modules=nd4j-api,nd4j-native-api,nd4j-presets-common,nd4j-native-preset,nd4j-cpu-backend-common,nd4j-native" \
    "target_modules=tokenizers-native-preset,tokenizers-native,nd4j-ggml,samediff-llm,nd4j-sdx-preset,nd4j-sdx-model,nd4j-sdx,sdx-aot" \
    "dependency_scope=runtime" \
    "compiler_goal=compiler:compile" \
    "${maven_compile_flags[@]}" |
    sha256sum | cut -d ' ' -f 1
)"

# Compile the complete managed CPU runtime in one reactor. This includes the
# JavaCPP preset hierarchy used by the generated Nd4jCpu binding, not only the
# backend factory hierarchy. Every direct source-owned dependency therefore
# comes from one receipt-bound tree instead of a mixture of fresh classes and
# cached Maven snapshots.
env JAVA_HOME="$JAVA_HOME_ARG" PATH="$JAVA_HOME_ARG/bin:$PATH" \
  "$MAVEN" "${maven_compile_flags[@]}" -f "$DL4J_ROOT/pom.xml" \
    -Pcpu-managed -pl :nd4j-api,:nd4j-native-api,:nd4j-presets-common,:nd4j-native-preset,:nd4j-cpu-backend-common,:nd4j-native \
    -Dmaven.compiler.outputDirectory="$NATIVE_RUNTIME_CLASSES" \
    -Dnd4j.api.outputDirectory="$NATIVE_RUNTIME_CLASSES" \
    -Dnd4j.nativeApi.outputDirectory="$NATIVE_RUNTIME_CLASSES" \
    -Dnd4j.cpuBackendCommon.outputDirectory="$NATIVE_RUNTIME_CLASSES" \
    compiler:compile
for module_id in nd4j-api nd4j-native-api nd4j-presets-common nd4j-native-preset nd4j-cpu-backend-common nd4j-native; do
  [[ -s "$NATIVE_RUNTIME_CLASSES/${MODULE_PROBE_CLASSES[$module_id]}" ]] ||
    fail "fresh Maven reactor compilation omitted ${MODULE_PROBE_CLASSES[$module_id]} for $module_id"
done

for module_id in tokenizers-native-preset tokenizers-native nd4j-ggml samediff-llm nd4j-sdx-preset nd4j-sdx-model nd4j-sdx sdx-aot; do
  module_root="$DL4J_ROOT/${MODULE_ROOTS[$module_id]}"
  module_classes="${MODULE_CLASS_DIRS[$module_id]}"
  mkdir -p "$module_classes"
  env JAVA_HOME="$JAVA_HOME_ARG" PATH="$JAVA_HOME_ARG/bin:$PATH" \
    "$MAVEN" "${maven_compile_flags[@]}" -f "$module_root/pom.xml" \
      -Dmaven.compiler.outputDirectory="$module_classes" compiler:compile
  [[ -s "$module_classes/${MODULE_PROBE_CLASSES[$module_id]}" ]] ||
    fail "fresh Maven compilation omitted ${MODULE_PROBE_CLASSES[$module_id]} for $module_id"
done

# The Android image must use the exact same official Hugging Face facade and
# JavaCPP tokenizer implementation as the desktop execution gate. These constant-
# pool checks are a build-time linkage assertion; they do not introduce reflection
# or an alternate tokenizer implementation into the runtime.
AOT_COMPILED_CORE="${MODULE_CLASS_DIRS[sdx-aot]}/org/eclipse/deeplearning4j/sdx/aot/SdxCompiledLlmCore.class"
HUGGING_FACE_TOKENIZER_CLASS="${MODULE_CLASS_DIRS[samediff-llm]}/org/eclipse/deeplearning4j/llm/tokenizer/HuggingFaceTokenizer.class"
NATIVE_TOKENIZER_CLASS="${MODULE_CLASS_DIRS[tokenizers-native]}/org/eclipse/deeplearning4j/tokenizers/NativeTokenizer.class"
[[ -s "$AOT_COMPILED_CORE" && -s "$HUGGING_FACE_TOKENIZER_CLASS" && -s "$NATIVE_TOKENIZER_CLASS" ]] ||
  fail "fresh Android compilation omitted the canonical tokenizer execution path"
grep -aFq 'org/eclipse/deeplearning4j/llm/tokenizer/HuggingFaceTokenizer' "$AOT_COMPILED_CORE" ||
  fail "SdxCompiledLlmCore is not linked directly to HuggingFaceTokenizer"
grep -aFq 'org/eclipse/deeplearning4j/tokenizers/NativeTokenizer' "$HUGGING_FACE_TOKENIZER_CLASS" ||
  fail "HuggingFaceTokenizer is not linked directly to the JavaCPP NativeTokenizer"

# compiler:compile intentionally avoids each module's target directory, so it
# does not execute process-resources. Merge the resources from every freshly
# compiled module into the same class directory that Native Image receives.
# This preserves services and Graal reachability metadata for the entire source
# closure instead of maintaining an error-prone allowlist of selected modules.
for module_id in "${FRESH_COMPILE_ORDER[@]}"; do
  module_resources="$DL4J_ROOT/${MODULE_ROOTS[$module_id]}/src/main/resources"
  sdx_stage_module_resources \
    "$module_resources" "${MODULE_CLASS_DIRS[$module_id]}" "$module_id" ||
    fail "could not stage resources for $module_id"
done

ND4J_API_NATIVE_IMAGE_ROOT="$NATIVE_RUNTIME_CLASSES/META-INF/native-image/org.eclipse.deeplearning4j/nd4j-api"
ND4J_API_REFLECT_CONFIG="$ND4J_API_NATIVE_IMAGE_ROOT/reflect-config.json"
ND4J_API_NATIVE_IMAGE_PROPERTIES="$ND4J_API_NATIVE_IMAGE_ROOT/native-image.properties"
[[ -s "$ND4J_API_REFLECT_CONFIG" ]] ||
  fail "fresh Android classpath omitted nd4j-api reflection metadata"
[[ -s "$ND4J_API_NATIVE_IMAGE_PROPERTIES" ]] ||
  fail "fresh Android classpath omitted nd4j-api Native Image configuration"
grep -F -q '"name": "org.nd4j.linalg.api.ops.DynamicCustomOp"' "$ND4J_API_REFLECT_CONFIG" ||
  fail "nd4j-api reflection metadata omitted DynamicCustomOp"
grep -F -q -- '--features=org.nd4j.nativeimage.Nd4jOpsReflectionFeature' \
  "$ND4J_API_NATIVE_IMAGE_PROPERTIES" ||
  fail "nd4j-api Native Image configuration omitted the operation reflection feature"
for runtime_initialized_class in \
  org.nd4j.linalg.api.ops \
  org.nd4j.autodiff.samediff \
  'org.bytedeco.javacpp.Loader$Helper'; do
  grep -F -q -- "$runtime_initialized_class" "$ND4J_API_NATIVE_IMAGE_PROPERTIES" ||
    fail "nd4j-api Native Image configuration did not defer $runtime_initialized_class until runtime"
done

ND4J_NATIVE_IMAGE_PROPERTIES="$NATIVE_RUNTIME_CLASSES/META-INF/native-image/org.eclipse.deeplearning4j/nd4j-native/native-image.properties"
[[ -s "$ND4J_NATIVE_IMAGE_PROPERTIES" ]] ||
  fail "fresh Android classpath omitted nd4j-native Native Image configuration"
for runtime_initialized_class in \
  org.nd4j.linalg.cpu.nativecpu.CpuEnvironment \
  'org.nd4j.linalg.cpu.nativecpu.bindings.Nd4jCpu$Environment'; do
  grep -F -q -- "$runtime_initialized_class" "$ND4J_NATIVE_IMAGE_PROPERTIES" ||
    fail "nd4j-native Native Image configuration did not defer $runtime_initialized_class until runtime"
done

CLASSES_DIR="${MODULE_CLASS_DIRS[sdx-aot]}"
MODEL_CLASSES_DIR="${MODULE_CLASS_DIRS[nd4j-sdx-model]}"
declare -A CURRENT_CLASSPATH_DIRS=(
  [nd4j-native-runtime]="$NATIVE_RUNTIME_CLASSES"
  [tokenizers-native-preset]="$FRESH_CLASSES_ROOT/tokenizers-native-preset"
  [tokenizers-native]="$FRESH_CLASSES_ROOT/tokenizers-native"
  [nd4j-ggml]="$FRESH_CLASSES_ROOT/nd4j-ggml"
  [samediff-llm]="$FRESH_CLASSES_ROOT/samediff-llm"
  [nd4j-sdx]="$FRESH_CLASSES_ROOT/nd4j-sdx"
  [nd4j-sdx-model]="$FRESH_CLASSES_ROOT/nd4j-sdx-model"
  [nd4j-sdx-preset]="$FRESH_CLASSES_ROOT/nd4j-sdx-preset"
)
CURRENT_CLASSPATH_IDS=(nd4j-native-runtime tokenizers-native-preset tokenizers-native nd4j-ggml samediff-llm nd4j-sdx nd4j-sdx-model nd4j-sdx-preset)
OVERRIDDEN_CLASSPATH_ARTIFACT_IDS=(
  nd4j-api
  nd4j-native-api
  nd4j-presets-common
  nd4j-native-preset
  nd4j-cpu-backend-common
  nd4j-native
  tokenizers-native-preset
  tokenizers-native
  nd4j-ggml
  samediff-llm
  nd4j-sdx
  nd4j-sdx-model
  nd4j-sdx-preset
)
# Android uses the OpenBLAS closure supplied by the attested CPU base SDK. The
# generic JavaCPP MKL artifact is a host-side alternative, not a fallback for
# the embedded image, and its static initializers may probe unavailable symbols.
FORBIDDEN_ANDROID_ARTIFACT_IDS=(mkl)
[[ -s "$MODEL_CLASSES_DIR/org/nd4j/dsp/model/SdxLlmNative.class" ]] ||
  fail "fresh compilation omitted SdxLlmNative.class"
CLASSPATH_FILE="$BUILD_ROOT/resolved-classpath.txt"
env JAVA_HOME="$JAVA_HOME_ARG" PATH="$JAVA_HOME_ARG/bin:$PATH" \
  "$MAVEN" "${maven_compile_flags[@]}" -f "$MODULE_DIR/pom.xml" \
    dependency:build-classpath -Dmdep.includeScope=runtime -Dmdep.outputFile="$CLASSPATH_FILE"
[[ -s "$CLASSPATH_FILE" ]] || fail "Maven did not resolve the sdx-aot runtime classpath"

EFFECTIVE_CLASSPATH_FILE="$BUILD_ROOT/effective-classpath.txt"
effective_classpath_entries=()
for current_id in "${CURRENT_CLASSPATH_IDS[@]}"; do
  effective_classpath_entries+=("${CURRENT_CLASSPATH_DIRS[$current_id]}")
done
while IFS= read -r classpath_entry; do
  [[ -n "$classpath_entry" ]] || continue
  skip_classpath_entry=0
  classpath_basename="$(basename -- "$classpath_entry")"
  classpath_artifact_id=""
  if [[ -f "$classpath_entry" && "$classpath_basename" == *.jar ]]; then
    candidate_artifact_id="$(basename -- "$(dirname -- "$(dirname -- "$classpath_entry")")")"
    if [[ "$classpath_basename" == "$candidate_artifact_id-"*.jar ]]; then
      classpath_artifact_id="$candidate_artifact_id"
    fi
  fi
  for current_id in "${CURRENT_CLASSPATH_IDS[@]}"; do
    if [[ "$classpath_entry" == "${CURRENT_CLASSPATH_DIRS[$current_id]}" ]]; then
      skip_classpath_entry=1
      break
    fi
  done
  if [[ "$skip_classpath_entry" == "0" ]]; then
    for artifact_id in "${OVERRIDDEN_CLASSPATH_ARTIFACT_IDS[@]}"; do
      if [[ "$classpath_artifact_id" == "$artifact_id" ]]; then
        skip_classpath_entry=1
        break
      fi
    done
  fi
  if [[ "$skip_classpath_entry" == "0" ]]; then
    for artifact_id in "${FORBIDDEN_ANDROID_ARTIFACT_IDS[@]}"; do
      if [[ "$classpath_artifact_id" == "$artifact_id" ]]; then
        skip_classpath_entry=1
        break
      fi
    done
  fi
  # Platform classifier jars only carry host native resources. This image gets
  # its Android native closure from the independently attested base SDK.
  if [[ "$skip_classpath_entry" == "0" &&
        "$classpath_basename" =~ -(linux|windows|macosx|ios)-[A-Za-z0-9_]+([.]jar)$ ]]; then
    skip_classpath_entry=1
  fi
  [[ "$skip_classpath_entry" == "1" ]] || effective_classpath_entries+=("$classpath_entry")
done < <(tr ':' '\n' <"$CLASSPATH_FILE")
(
  IFS=:
  printf '%s\n' "${effective_classpath_entries[*]}"
) >"$EFFECTIVE_CLASSPATH_FILE"

# JavaCPP reaches generated backend classes through both Class.forName and JNI.
# Derive the complete recursive binding closure from the exact cross-target
# classpath. The expected roots below are fail-closed product assertions only;
# they are not the registration source of truth.
JAVACPP_REACHABILITY_GENERATOR="$SCRIPT_DIR/JavaCppNativeImageReachability.java"
[[ -s "$JAVACPP_REACHABILITY_GENERATOR" ]] ||
  fail "JavaCPP Native Image reachability generator is missing"
JAVACPP_REACHABILITY_GENERATOR_SHA256="$(sha256_file "$JAVACPP_REACHABILITY_GENERATOR")"
JAVACPP_REACHABILITY_CONFIG_DIR="$CLASSES_DIR/META-INF/native-image/org.eclipse.deeplearning4j/sdx-aot-javacpp"
JAVACPP_REACHABILITY_CONFIG="$JAVACPP_REACHABILITY_CONFIG_DIR/reflect-config.json"
JAVACPP_JNI_CONFIG="$JAVACPP_REACHABILITY_CONFIG_DIR/jni-config.json"
JAVACPP_INITIALIZATION_CONFIG="$JAVACPP_REACHABILITY_CONFIG_DIR/native-image.properties"
JAVACPP_REACHABILITY_MANIFEST="$METADATA_DIR/javacpp-native-image-reachability.txt"
EXPECTED_JAVACPP_BINDING_ROOTS=(
  org.nd4j.linalg.cpu.nativecpu.bindings.Nd4jCpu
  org.nd4j.dsp.model.SdxLlmNative
  org.eclipse.deeplearning4j.tokenizers.bindings.TokenizersNative
)
EXPECTED_JAVACPP_NESTED_BINDINGS=(
  'org.nd4j.linalg.cpu.nativecpu.bindings.Nd4jCpu$Environment'
  'org.nd4j.linalg.cpu.nativecpu.bindings.Nd4jCpu$ConstNDArrayVector$Iterator'
)
env JAVA_HOME="$JAVA_HOME_ARG" PATH="$JAVA_HOME_ARG/bin:$PATH" \
  "$JAVA_HOME_ARG/bin/java" \
    -Dorg.bytedeco.javacpp.platform=android-arm64 \
    "$JAVACPP_REACHABILITY_GENERATOR" \
    "$EFFECTIVE_CLASSPATH_FILE" \
    "$JAVACPP_REACHABILITY_CONFIG" \
    "$JAVACPP_JNI_CONFIG" \
    "$JAVACPP_INITIALIZATION_CONFIG" \
    "$JAVACPP_REACHABILITY_MANIFEST"
[[ -s "$JAVACPP_REACHABILITY_CONFIG" &&
   -s "$JAVACPP_JNI_CONFIG" &&
   -s "$JAVACPP_INITIALIZATION_CONFIG" &&
   -s "$JAVACPP_REACHABILITY_MANIFEST" ]] ||
  fail "JavaCPP Native Image reachability metadata is empty"
for binding_root in "${EXPECTED_JAVACPP_BINDING_ROOTS[@]}"; do
  grep -Fqx "root=$binding_root" "$JAVACPP_REACHABILITY_MANIFEST" ||
    fail "JavaCPP reachability metadata omits binding root $binding_root"
done
for binding_class in "${EXPECTED_JAVACPP_NESTED_BINDINGS[@]}"; do
  grep -Fqx "reflection-class=$binding_class" "$JAVACPP_REACHABILITY_MANIFEST" ||
    fail "JavaCPP reflection metadata omits nested binding $binding_class"
  grep -Fqx "jni-class=$binding_class" "$JAVACPP_REACHABILITY_MANIFEST" ||
    fail "JavaCPP JNI metadata omits nested binding $binding_class"
done
EXPECTED_OPENBLAS_GLOBAL=org.bytedeco.openblas.global.openblas_nolapack
grep -Fqx "reflection-class=$EXPECTED_OPENBLAS_GLOBAL" "$JAVACPP_REACHABILITY_MANIFEST" ||
  fail "Android CPU JavaCPP reflection metadata does not resolve $EXPECTED_OPENBLAS_GLOBAL"
grep -Fqx "jni-class=$EXPECTED_OPENBLAS_GLOBAL" "$JAVACPP_REACHABILITY_MANIFEST" ||
  fail "Android CPU JavaCPP JNI metadata does not resolve $EXPECTED_OPENBLAS_GLOBAL"
JAVACPP_REACHABILITY_CONFIG_SHA256="$(sha256_file "$JAVACPP_REACHABILITY_CONFIG")"
JAVACPP_JNI_CONFIG_SHA256="$(sha256_file "$JAVACPP_JNI_CONFIG")"
JAVACPP_INITIALIZATION_CONFIG_SHA256="$(sha256_file "$JAVACPP_INITIALIZATION_CONFIG")"
JAVACPP_REACHABILITY_MANIFEST_SHA256="$(sha256_file "$JAVACPP_REACHABILITY_MANIFEST")"

# Native Image reads this classpath configuration itself. Keep production's
# historical/default O2 class tree byte-for-byte stable, while dev builds add
# the documented fastest-build optimization flag and therefore get a distinct
# object-cache identity.
if [[ "$NATIVE_IMAGE_BUILD_MODE" == dev ]]; then
  NATIVE_IMAGE_OPTIMIZATION_CONFIG_DIR="$CLASSES_DIR/META-INF/native-image/org.eclipse.deeplearning4j/sdx-aot-build"
  NATIVE_IMAGE_OPTIMIZATION_CONFIG="$NATIVE_IMAGE_OPTIMIZATION_CONFIG_DIR/native-image.properties"
  mkdir -p "$NATIVE_IMAGE_OPTIMIZATION_CONFIG_DIR"
  printf '%s\n' 'Args = -Ob' >"$NATIVE_IMAGE_OPTIMIZATION_CONFIG"
  [[ "$(sha256_file "$NATIVE_IMAGE_OPTIMIZATION_CONFIG")" == "$NATIVE_IMAGE_OPTIMIZATION_CONFIG_SHA256" ]] ||
    fail "Native Image quick-build configuration changed while staging"
fi
cat >"$METADATA_DIR/native-image-optimization.txt" <<OPTIMIZATION
build_mode=$NATIVE_IMAGE_BUILD_MODE
optimization=-O$NATIVE_IMAGE_OPTIMIZATION
config_sha256=$NATIVE_IMAGE_OPTIMIZATION_CONFIG_SHA256
OPTIMIZATION

# Hash only after generated Native Image metadata has joined the classes tree.
CLASSES_SHA256="$(tree_manifest_sha256 "$CLASSES_DIR")"
MODEL_CLASSES_SHA256="$(tree_manifest_sha256 "$MODEL_CLASSES_DIR")"

FRESH_CLASS_BUILDS="$METADATA_DIR/fresh-class-builds.txt"
for module_id in "${FRESH_COMPILE_ORDER[@]}"; do
  printf '%s %s %s\n' \
    "$module_id" \
    "$(module_source_manifest_sha256 "${MODULE_ROOTS[$module_id]}")" \
    "$(tree_manifest_sha256 "${MODULE_CLASS_DIRS[$module_id]}")"
done >"$FRESH_CLASS_BUILDS"
FRESH_CLASS_BUILDS_SHA256="$(sha256_file "$FRESH_CLASS_BUILDS")"

CLASSPATH_MANIFEST="$METADATA_DIR/classpath-bytes.txt"
classpath_index=0
while IFS= read -r classpath_entry; do
  [[ -n "$classpath_entry" ]] || continue
  classpath_index=$((classpath_index + 1))
  # Cache identity is the ordered kind+bytes sequence. Absolute build and Maven
  # paths are deliberately excluded: the disposable generation directory changes
  # on every retry even when the exact Native Image input is unchanged.
  classpath_label="$(basename -- "$classpath_entry")"
  if [[ -f "$classpath_entry" ]]; then
    printf '%06d file %s %s\n' "$classpath_index" "$(sha256_file "$classpath_entry")" "$classpath_label"
  elif [[ -d "$classpath_entry" ]]; then
    printf '%06d tree %s %s\n' "$classpath_index" "$(tree_manifest_sha256 "$classpath_entry")" "$classpath_label"
  else
    fail "classpath entry is missing: $classpath_entry"
  fi
done < <(tr ':' '\n' <"$EFFECTIVE_CLASSPATH_FILE") >"$CLASSPATH_MANIFEST"
[[ "$classpath_index" -gt 0 ]] || fail "classpath is empty"
CLASSPATH_MANIFEST_SHA256="$(sha256_file "$CLASSPATH_MANIFEST")"

# Fresh class directories are represented by the target-scoped source closure.
# Hash resolved runtime files separately by bytes and in classpath order so
# SNAPSHOT names, timestamps, and Maven coordinates can never masquerade as an
# unchanged Native Image dependency.
RUNTIME_DEPENDENCY_MANIFEST="$METADATA_DIR/runtime-dependency-bytes.txt"
runtime_dependency_index=0
while IFS= read -r classpath_entry; do
  [[ -n "$classpath_entry" ]] || continue
  if [[ -f "$classpath_entry" ]]; then
    runtime_dependency_index=$((runtime_dependency_index + 1))
    printf '%06d %s %s\n' \
      "$runtime_dependency_index" \
      "$(sha256_file "$classpath_entry")" \
      "$(basename -- "$classpath_entry")"
  elif [[ ! -d "$classpath_entry" ]]; then
    fail "runtime classpath entry is missing: $classpath_entry"
  fi
done < <(tr ':' '\n' <"$EFFECTIVE_CLASSPATH_FILE") >"$RUNTIME_DEPENDENCY_MANIFEST"
[[ "$runtime_dependency_index" -gt 0 ]] || fail "resolved runtime dependency classpath is empty"
RUNTIME_DEPENDENCY_MANIFEST_SHA256="$(sha256_file "$RUNTIME_DEPENDENCY_MANIFEST")"
RUNTIME_ANALYSIS_MANIFEST="$METADATA_DIR/runtime-analysis-bytes.txt"
sdx_native_image_runtime_analysis_manifest "$EFFECTIVE_CLASSPATH_FILE" >"$RUNTIME_ANALYSIS_MANIFEST" ||
  fail "could not fingerprint the managed Native Image dependency closure"
RUNTIME_ANALYSIS_MANIFEST_SHA256="$(sha256_file "$RUNTIME_ANALYSIS_MANIFEST")"

# The managed closure is cross-target input, not a host JVM classpath. Reject
# host classifier jars and any embedded MKL type before Native Image can make a
# host-only static initializer reachable on Android.
if grep -Eq -- '-(linux|windows|macosx|ios)-[A-Za-z0-9_]+[.]jar$' "$CLASSPATH_MANIFEST"; then
  fail "Android AOT classpath contains a non-Android JavaCPP classifier"
fi
if grep -Eq -- '[[:space:]]mkl-[^[:space:]]*[.]jar$' "$CLASSPATH_MANIFEST"; then
  fail "Android AOT classpath contains the JavaCPP MKL artifact"
fi
FORBIDDEN_CLASS_AUDIT="$BUILD_ROOT/forbidden-android-classpath.txt"
env JAVA_HOME="$JAVA_HOME_ARG" PATH="$JAVA_HOME_ARG/bin:$PATH" \
  "$JAVA_HOME_ARG/bin/java" "$SCRIPT_DIR/EmbeddedClasspathAudit.java" \
    "$EFFECTIVE_CLASSPATH_FILE" 'org/bytedeco/mkl/global/mkl_rt' >"$FORBIDDEN_CLASS_AUDIT"
if [[ -s "$FORBIDDEN_CLASS_AUDIT" ]]; then
  IFS= read -r forbidden_class <"$FORBIDDEN_CLASS_AUDIT"
  fail "Android AOT classpath contains a host-only MKL class: $forbidden_class"
fi

OBJECT_STAGE_FORMAT="$SDX_NATIVE_IMAGE_OBJECT_STAGE_FORMAT"
OBJECT_CACHE_SCHEMA="$SDX_NATIVE_IMAGE_OBJECT_CACHE_SCHEMA"
OBJECT_CONTRACT="$SDX_NATIVE_IMAGE_OBJECT_CONTRACT"
OBJECT_CACHE_TARGET="$SDX_NATIVE_IMAGE_OBJECT_CACHE_TARGET"
OBJECT_CACHE_ARTIFACT="$SDX_NATIVE_IMAGE_OBJECT_CACHE_ARTIFACT"

# Cache the completed Android relocatable object by managed analysis inputs.
# Native payload bytes and final link inputs are deliberately excluded: they
# cannot change Graal reachability and are bound separately by the final SDK
# receipt. A real managed source, metadata, toolchain, or build-mode change
# still produces a distinct checksum-verified object.
OBJECT_STAGE_INPUTS_SHA256="$(sdx_native_image_object_identity_sha256)" ||
  fail "could not calculate the Native Image object identity"
OBJECT_STAGE_DIR="$OBJECT_STAGES_DIR/$OBJECT_STAGE_INPUTS_SHA256"
printf 'Native Image managed-object cache key: %s\n' "$OBJECT_STAGE_INPUTS_SHA256"

if [[ "$FRESH_CLASSES_ONLY" == 1 ]]; then
  printf 'Compiled and preserved the exact SDX Android AOT classpath.\n'
  printf '  build root: %s\n' "$BUILD_ROOT"
  printf '  classes:    %s\n' "$FRESH_CLASSES_ROOT"
  printf '  classpath:  %s\n' "$EFFECTIVE_CLASSPATH_FILE"
  exit 0
fi

validate_native_image_object() {
  local object="$1"
  local header="$2"
  [[ -f "$object" && ! -L "$object" && -s "$object" ]] || return 1
  "$LLVM_READELF" -h "$object" >"$header" || return 1
  grep -q 'Machine:.*AArch64' "$header"
}

validate_native_image_object_stage() {
  local stage="$1"
  local receipt="$stage/build-receipt"
  local object="$stage/libsdx_llm.o"
  local header="$BUILD_ROOT/cached-libsdx_llm.o.elf-header"
  local object_sha256
  [[ -d "$stage" && ! -L "$stage" ]] || return 1
  ! find "$stage" -type l -print -quit | grep -q . || return 1
  ! find "$stage" -perm /0222 -print -quit | grep -q . || return 1
  [[ -f "$receipt" && ! -L "$receipt" && -s "$receipt" ]] || return 1
  while IFS= read -r expected; do
    receipt_has "$receipt" "$expected" || return 1
  done < <(sdx_native_image_object_identity_lines)
  receipt_has "$receipt" "object_stage_inputs_sha256=$OBJECT_STAGE_INPUTS_SHA256" || return 1
  validate_native_image_object "$object" "$header" || return 1
  object_sha256="$(sha256_file "$object")"
  receipt_has "$receipt" "object_sha256=$object_sha256"
}

# A final-link or JNI transport edit changes the full SDK source receipt but not
# Graal's managed analysis closure. Accept an older object-stage key only when
# every analysis-affecting identity line still matches; the broad source manifest
# and its derived stage key are intentionally the only exceptions.
validate_compatible_native_image_object_stage() {
  local stage="$1"
  local receipt="$stage/build-receipt"
  local object="$stage/libsdx_llm.o"
  local header="$BUILD_ROOT/compatible-libsdx_llm.o.elf-header"
  local expected object_sha256
  [[ -d "$stage" && ! -L "$stage" ]] || return 1
  ! find "$stage" -type l -print -quit | grep -q . || return 1
  ! find "$stage" -perm /0222 -print -quit | grep -q . || return 1
  [[ -f "$receipt" && ! -L "$receipt" && -s "$receipt" ]] || return 1
  while IFS= read -r expected; do
    case "$expected" in
      source_manifest_sha256=*) continue ;;
    esac
    receipt_has "$receipt" "$expected" || return 1
  done < <(sdx_native_image_object_identity_lines)
  validate_native_image_object "$object" "$header" || return 1
  object_sha256="$(sha256_file "$object")"
  receipt_has "$receipt" "object_sha256=$object_sha256"
}

stage_native_image_object() {
  local source_stage="$1"
  cp --reflink=auto -- "$source_stage/libsdx_llm.o" "$OBJECT"
  [[ "$(stat -c '%d:%i' "$source_stage/libsdx_llm.o")" != "$(stat -c '%d:%i' "$OBJECT")" ]] ||
    fail "cached Native Image object was hard-linked into the build"
  OBJECT_SHA256="$(sha256_file "$OBJECT")"
  [[ "$OBJECT_SHA256" == "$(sha256_file "$source_stage/libsdx_llm.o")" ]] ||
    fail "cached Native Image object changed while staging"
}

OBJECT_REUSED=0
if [[ "$SDX_NATIVE_CACHE" == 1 && "$SDX_NATIVE_FORCE_REBUILD" != 1 ]]; then
  if validate_native_image_object_stage "$OBJECT_STAGE_DIR"; then
    stage_native_image_object "$OBJECT_STAGE_DIR"
    printf 'CACHE HIT: reusing validated local Native Image object stage: %s\n' "$OBJECT_STAGE_INPUTS_SHA256"
    OBJECT_REUSED=1
  elif [[ -e "$OBJECT_STAGE_DIR" ]]; then
    fail "Native Image object stage exists but failed validation: $OBJECT_STAGE_DIR"
  else
    for compatible_stage in "$OBJECT_STAGES_DIR"/*; do
      [[ -d "$compatible_stage" ]] || continue
      if validate_compatible_native_image_object_stage "$compatible_stage"; then
        stage_native_image_object "$compatible_stage"
        printf 'CACHE HIT: reusing analysis-compatible Native Image object stage: %s\n' \
          "$(basename -- "$compatible_stage")"
        OBJECT_REUSED=1
        break
      fi
    done
    if [[ "$OBJECT_REUSED" == 0 ]] && sdx_native_cache_restore \
        "$OBJECT_CACHE_TARGET" "$OBJECT_STAGE_INPUTS_SHA256" "$OBJECT_CACHE_ARTIFACT" "$OBJECT"; then
      OBJECT_ELF_HEADER="$BUILD_ROOT/shared-cache-libsdx_llm.o.elf-header"
      validate_native_image_object "$OBJECT" "$OBJECT_ELF_HEADER" ||
        fail "shared cached Native Image object is not Android AArch64"
      OBJECT_SHA256="$(sha256_file "$OBJECT")"
      printf 'Reusing checksum-verified shared Native Image object: %s\n' "$OBJECT_STAGE_INPUTS_SHA256"
      OBJECT_REUSED=1
    fi
  fi
fi

if [[ "$OBJECT_REUSED" == 0 ]]; then
  if [[ "$SDX_NATIVE_CACHE" == 0 ]]; then
    printf 'Native Image object cache disabled; rebuilding %s.\n' "$OBJECT_STAGE_INPUTS_SHA256"
  elif [[ "$SDX_NATIVE_FORCE_REBUILD" == 1 ]]; then
    printf 'Native Image object cache bypassed by force rebuild: %s\n' "$OBJECT_STAGE_INPUTS_SHA256"
  else
    printf 'CACHE MISS: Native Image object %s\n' "$OBJECT_STAGE_INPUTS_SHA256"
  fi
  object_args=(
    --android-ndk "$ANDROID_NDK"
    --graalvm-home "$GRAALVM_HOME"
    --jobs "$JOBS"
    --work-dir "$OBJECT_WORK"
    --output-dir "$BUILD_ROOT/unused-graph-sdk"
    --classes-dir "$CLASSES_DIR"
    --classpath-file "$EFFECTIVE_CLASSPATH_FILE"
    --strict-classpath
    --reuse-jdk-libs "$REUSE_JDK_LIBS"
    --reuse-svm-libs "$REUSE_SVM_LIBS"
    --object-output "$OBJECT"
  )
  [[ "$OFFLINE" == "0" ]] || object_args+=(--offline)
  "$OBJECT_BUILDER" "${object_args[@]}"
  OBJECT_ELF_HEADER="$BUILD_ROOT/libsdx_llm.o.elf-header"
  validate_native_image_object "$OBJECT" "$OBJECT_ELF_HEADER" ||
    fail "Native Image relocatable object was not produced as an Android AArch64 object"
  OBJECT_SHA256="$(sha256_file "$OBJECT")"

  if [[ "$SDX_NATIVE_CACHE" == 1 ]]; then
    OBJECT_STAGE_TMP="$(mktemp -d "$OBJECT_STAGES_DIR/.native-image-object.XXXXXXXX")"
    cp --reflink=auto -- "$OBJECT" "$OBJECT_STAGE_TMP/libsdx_llm.o"
    [[ "$(stat -c '%d:%i' "$OBJECT")" != "$(stat -c '%d:%i' "$OBJECT_STAGE_TMP/libsdx_llm.o")" ]] ||
      fail "Native Image object stage used a mutable hard link"
    [[ "$(sha256_file "$OBJECT_STAGE_TMP/libsdx_llm.o")" == "$OBJECT_SHA256" &&
       "$(sha256_file "$OBJECT")" == "$OBJECT_SHA256" ]] ||
      fail "Native Image object changed while publishing its independent stage"
    {
      sdx_native_image_object_identity_lines
      printf '%s\n' \
        "object_stage_inputs_sha256=$OBJECT_STAGE_INPUTS_SHA256" \
        "runtime_dependency_manifest_sha256=$RUNTIME_DEPENDENCY_MANIFEST_SHA256" \
        "classes_sha256=$CLASSES_SHA256" \
        "model_classes_sha256=$MODEL_CLASSES_SHA256" \
        "fresh_class_builds_sha256=$FRESH_CLASS_BUILDS_SHA256" \
        "classpath_manifest_sha256=$CLASSPATH_MANIFEST_SHA256" \
        "object_sha256=$OBJECT_SHA256"
    } >"$OBJECT_STAGE_TMP/build-receipt"
    chmod -R a-w "$OBJECT_STAGE_TMP"
    if [[ -e "$OBJECT_STAGE_DIR" ]]; then
      validate_native_image_object_stage "$OBJECT_STAGE_DIR" ||
        fail "concurrently published Native Image object stage failed validation"
      chmod -R u+w "$OBJECT_STAGE_TMP"
      rm -rf -- "$OBJECT_STAGE_TMP"
    else
      mv -- "$OBJECT_STAGE_TMP" "$OBJECT_STAGE_DIR"
    fi
    printf 'Published local Native Image object stage: %s\n' "$OBJECT_STAGE_INPUTS_SHA256"
  fi
fi

OBJECT_ELF_HEADER="$BUILD_ROOT/libsdx_llm.o.elf-header"
validate_native_image_object "$OBJECT" "$OBJECT_ELF_HEADER" ||
  fail "staged Native Image object is not Android AArch64"
[[ "$(sha256_file "$OBJECT")" == "$OBJECT_SHA256" ]] ||
  fail "Native Image object changed after stage validation"
if [[ "$SDX_NATIVE_CACHE" == 1 ]]; then
  sdx_native_cache_publish \
    "$OBJECT_CACHE_TARGET" "$OBJECT_STAGE_INPUTS_SHA256" "$OBJECT_CACHE_ARTIFACT" "$OBJECT" ||
    printf 'WARNING: could not publish shared Native Image object cache entry %s\n' \
      "$OBJECT_STAGE_INPUTS_SHA256" >&2
fi

UNSTRIPPED="$BUILD_ROOT/libsdx_llm.unstripped.so"
LINKER_SCRIPT="$MODULE_DIR/src/main/linker/sdx_exports.lds"
[[ -s "$LINKER_SCRIPT" ]] || fail "SDX export version script is missing"
"$CLANG" -shared -o "$UNSTRIPPED" "$OBJECT"   -Wl,--start-group     "$REUSE_SVM_LIBS/libjvm.a" "$REUSE_SVM_LIBS/liblibchelper.a"     "$REUSE_JDK_LIBS/libjava.a" "$REUSE_JDK_LIBS/libnet.a"     "$REUSE_JDK_LIBS/libnio.a" "$REUSE_JDK_LIBS/libzip.a"     "$REUSE_JDK_LIBS/libprefs.a" "$REUSE_JDK_LIBS/libextnet.a"   -Wl,--end-group   -ldl -lz -lm -llog   -Wl,--no-undefined -Wl,--gc-sections -Wl,--build-id=sha1   -Wl,-z,relro,-z,now -Wl,-z,max-page-size=16384 -Wl,-z,common-page-size=16384   -Wl,-soname,libsdx_llm.so -Wl,--version-script="$LINKER_SCRIPT"
# JavaCPP registers JNI names independently of Java reachability, so the raw
# Loader.Helper.addressof name can remain as inert metadata. Require the image-wide
# substitutions instead: both Loader entry points throw this diagnostic before any
# host-VM JNIEnv can be reached, regardless of which backend or initializer calls them.
JAVACPP_ADDRESSOF_GUARD="Embedded native images must resolve process symbols through NativeOps, not Loader.addressof: "
LC_ALL=C grep -aFq "$JAVACPP_ADDRESSOF_GUARD" "$UNSTRIPPED" ||
  fail "libsdx_llm.so omitted the embedded-runtime JavaCPP Loader.addressof guard"
cp "$UNSTRIPPED" "$JNI_DIR/libsdx_llm.so"
"$LLVM_STRIP" --strip-unneeded "$JNI_DIR/libsdx_llm.so"
assert_unix_file_attributes_abi "$JNI_DIR/libsdx_llm.so"
LIBSDX_DYNAMIC_SYMBOLS="$BUILD_ROOT/libsdx_llm.dynamic-symbols"
"$LLVM_NM" -D --defined-only "$JNI_DIR/libsdx_llm.so" >"$LIBSDX_DYNAMIC_SYMBOLS"
for symbol in sdxLlmAbiVersion sdxLlmPrepareGguf sdxLlmResolveModelBundle sdxLlmLoadCompiledModel sdxLlmGenerateStreaming sdxLlmParseChatResult; do
  grep -q "[[:space:]]${symbol}@@SDX_LLM_1$" "$LIBSDX_DYNAMIC_SYMBOLS" ||
    fail "fresh libsdx_llm.so omitted required versioned export: ${symbol}@@SDX_LLM_1"
done

# The embedded image still needs JavaCPP's core JNI library for ND4J. Consumer
# transports are deliberately excluded from this SDK: Android applications own
# their JNI bridge to the stable sdx_llm_c.h ABI.
JAVACPP_LIFECYCLE_BRIDGE="$SCRIPT_DIR/javacpp_jni_lifecycle.cpp"
[[ -s "$JAVACPP_LIFECYCLE_BRIDGE" ]] || fail "JavaCPP JNI lifecycle bridge is missing"
"$GRAALVM_HOME/bin/java" -cp "$JAVACPP_JAR" org.bytedeco.javacpp.tools.Builder   -classpath "$MODEL_CLASSES_DIR" -d "$BRIDGE_DIR" -o jnisdx_llm -nocompile   org.nd4j.dsp.model.SdxLlmNative
[[ -s "$BRIDGE_DIR/jnijavacpp.cpp" ]] ||
  fail "JavaCPP did not generate the embedded-runtime core JNI source"

COMMON_BRIDGE_FLAGS=(
  -shared -fPIC -O2 -std=c++17 -DANDROID
  -Wl,--no-undefined -Wl,--build-id=sha1
  -Wl,-z,relro,-z,now -Wl,-z,max-page-size=16384 -Wl,-z,common-page-size=16384
)
"$CLANGXX" "${COMMON_BRIDGE_FLAGS[@]}"   -Wl,-soname,libjnijavacpp.so   "$BRIDGE_DIR/jnijavacpp.cpp" "$JAVACPP_LIFECYCLE_BRIDGE" -o "$JNI_DIR/libjnijavacpp.so"   -llog -ldl -lm
LIBJNIJAVACPP_DYNAMIC_SYMBOLS="$BUILD_ROOT/libjnijavacpp.dynamic-symbols"
"$LLVM_NM" -D --defined-only "$JNI_DIR/libjnijavacpp.so" >"$LIBJNIJAVACPP_DYNAMIC_SYMBOLS"
for symbol in JNI_OnLoad JNI_OnUnload JNI_OnLoad_jnijavacpp JNI_OnUnload_jnijavacpp; do
  grep -q "[[:space:]]${symbol}$" "$LIBJNIJAVACPP_DYNAMIC_SYMBOLS" ||
    fail "fresh libjnijavacpp.so omitted required lifecycle symbol: $symbol"
done

BASE_NATIVE_BYTES="$METADATA_DIR/base-sdk-native-bytes.txt"
while IFS= read -r library_name; do
  [[ "$library_name" =~ ^lib[A-Za-z0-9._+-]+[.]so$ ]] ||
    fail "unsafe base SDK native member: $library_name"
  case "$library_name" in
    libsdx_llm.so|libjnijavacpp.so|libc++_shared.so) continue ;;
  esac
  source_library="$BASE_SDK/jni/arm64-v8a/$library_name"
  [[ -f "$source_library" && ! -L "$source_library" && -s "$source_library" ]] ||
    fail "base SDK native member is missing, empty, or not a regular file: $library_name"
  source_mode="$(stat -c '%a' "$source_library")"
  (( (8#$source_mode & 0222) == 0 )) ||
    fail "base SDK native member is writable and therefore mutable: $source_library"
  source_sha256="$(sha256_file "$source_library")"
  target_library="$JNI_DIR/$library_name"
  cp --reflink=auto -- "$source_library" "$target_library"
  [[ "$(stat -c '%d:%i' "$source_library")" != "$(stat -c '%d:%i' "$target_library")" ]] ||
    fail "base SDK native member was hard-linked into the generation: $library_name"
  [[ "$(sha256_file "$target_library")" == "$source_sha256" &&
     "$(sha256_file "$source_library")" == "$source_sha256" ]] ||
    fail "base SDK native member changed during independent staging: $library_name"
  printf '%s %s\n' "$source_sha256" "$library_name"
done <"$BASE_SDK/metadata/cmake-owned-native-libraries.txt" | LC_ALL=C sort >"$BASE_NATIVE_BYTES"
cp "$CXX_SHARED" "$JNI_DIR/libc++_shared.so"

cp "$MODULE_DIR/include/sdx_llm_c.h" "$INCLUDE_DIR/sdx_llm_c.h"
cp "$JDK_SUPPORT_RECEIPT" "$METADATA_DIR/jdk-support-receipt"
cp "$BRIDGE_DIR/jnijavacpp.cpp" "$METADATA_DIR/jnijavacpp.cpp"
cp "$JAVACPP_LIFECYCLE_BRIDGE" "$METADATA_DIR/javacpp_jni_lifecycle.cpp"

NATIVE_MANIFEST="$METADATA_DIR/cmake-owned-native-libraries.txt"
find "$JNI_DIR" -maxdepth 1 -type f -name '*.so' -printf '%f\n' | LC_ALL=C sort -u >"$NATIVE_MANIFEST"
NATIVE_COUNT="$(wc -l <"$NATIVE_MANIFEST")"
[[ "$NATIVE_COUNT" -gt 3 ]] || fail "published SDK native closure is incomplete"
NATIVE_BYTES="$METADATA_DIR/sdk-native-bytes.txt"
while IFS= read -r library_name; do
  printf '%s %s\n' "$(sha256_file "$JNI_DIR/$library_name")" "$library_name"
done <"$NATIVE_MANIFEST" >"$NATIVE_BYTES"
SDK_NATIVE_BYTES_SHA256="$(sha256_file "$NATIVE_BYTES")"

CLOSURE="$METADATA_DIR/native-dependency-closure.txt"
: >"$CLOSURE"
while IFS= read -r library_name; do
  while IFS= read -r needed; do
    printf '%s -> %s\n' "$library_name" "$needed"
  done < <("$LLVM_READELF" -d "$JNI_DIR/$library_name" |
    sed -n 's/.*Shared library: \[\([^]]*\)\].*/\1/p' | LC_ALL=C sort -u)
done <"$NATIVE_MANIFEST" >"$CLOSURE"

BUILD_SCRIPT_SHA256="$(sha256_file "${BASH_SOURCE[0]}")"
NATIVE_IMAGE_CACHE_HELPER_SHA256="$(sha256_file "$NATIVE_IMAGE_CACHE_HELPER")"
MAVEN_SHA256="$(sha256_file "$MAVEN")"
MAVEN_VERSION="$({ env -u JAVA_TOOL_OPTIONS JAVA_HOME="$JAVA_HOME_ARG" PATH="$JAVA_HOME_ARG/bin:$PATH" "$MAVEN" --version; } 2>&1)"
MAVEN_VERSION_SHA256="$(printf '%s\n' "$MAVEN_VERSION" | sha256sum | cut -d ' ' -f 1)"
JAVA_VERSION="$({ env -u JAVA_TOOL_OPTIONS "$JAVA_HOME_ARG/bin/java" -version; } 2>&1)"
JAVA_VERSION_SHA256="$(printf '%s\n' "$JAVA_VERSION" | sha256sum | cut -d ' ' -f 1)"
BASE_SDK_NATIVE_SHA256="$(sha256_file "$BASE_NATIVE_BYTES")"
LINKER_SCRIPT_SHA256="$(sha256_file "$LINKER_SCRIPT")"
JAVACPP_JAR_SHA256="$(sha256_file "$JAVACPP_JAR")"
NDK_REVISION_SHA256="$(sha256_file "$ANDROID_NDK/source.properties")"
GRAALVM_VERSION_SHA256="$(printf '%s' "$NATIVE_IMAGE_VERSION" | sha256sum | cut -d ' ' -f 1)"
LIBSDX_SHA256="$(sha256_file "$JNI_DIR/libsdx_llm.so")"
LIBJNIJAVACPP_SHA256="$(sha256_file "$JNI_DIR/libjnijavacpp.so")"
JNIJAVACPP_SOURCE_SHA256="$(sha256_file "$METADATA_DIR/jnijavacpp.cpp")"
JAVACPP_LIFECYCLE_SOURCE_SHA256="$(sha256_file "$METADATA_DIR/javacpp_jni_lifecycle.cpp")"
NATIVE_MANIFEST_SHA256="$(sha256_file "$NATIVE_MANIFEST")"

[[ "$(sdx_git_source_manifest_sha256 "$DL4J_ROOT" "${DL4J_AOT_SOURCE_ROOTS[@]}")" == "$SOURCE_MANIFEST_SHA256" ]] ||
  fail "DL4J AOT source tree changed during the build"
if [[ -f "$BASE_SDK_INPUT" ]]; then
  [[ "$(sha256_file "$BASE_SDK_INPUT")" == "$BASE_SDK_INPUT_SHA256" ]] ||
    fail "base AAR changed during the build"
else
  [[ "$(tree_manifest_sha256 "$BASE_SDK_INPUT")" == "$BASE_SDK_INPUT_SHA256" ]] ||
    fail "base SDK changed during the build"
fi
for archive in libjava.a libnet.a libnio.a libzip.a libprefs.a libextnet.a; do
  [[ "$(sha256_file "$REUSE_JDK_LIBS_SOURCE/$archive")" == "${JDK_ARCHIVE_SHA256[$archive]}" &&
     "$(sha256_file "$LOCAL_JDK_LIBS/$archive")" == "${JDK_ARCHIVE_SHA256[$archive]}" ]] ||
    fail "JDK support archive changed during the build: $archive"
done
[[ "$(sha256_file "$REUSE_JDK_LIBS_SOURCE/jdk-support-receipt")" == "$JDK_SUPPORT_RECEIPT_SHA256" &&
   "$(sha256_file "$LOCAL_JDK_LIBS/jdk-support-receipt")" == "$JDK_SUPPORT_RECEIPT_SHA256" ]] ||
  fail "JDK support receipt changed during the build"
[[ "$(sha256_file "$REUSE_SVM_LIBS_SOURCE/libjvm.a")" == "$LIBJVM_SHA256" &&
   "$(sha256_file "$LOCAL_SVM_LIBS/libjvm.a")" == "$LIBJVM_SHA256" &&
   "$(sha256_file "$REUSE_SVM_LIBS_SOURCE/liblibchelper.a")" == "$LIBLIBCHELPER_SHA256" &&
   "$(sha256_file "$LOCAL_SVM_LIBS/liblibchelper.a")" == "$LIBLIBCHELPER_SHA256" ]] ||
  fail "SVM support archive changed during the build"

INPUTS_SHA256="$(
  printf '%s\n' \
    "source_manifest_sha256=$SOURCE_MANIFEST_SHA256" \
    "classes_sha256=$CLASSES_SHA256" \
    "model_classes_sha256=$MODEL_CLASSES_SHA256" \
    "fresh_class_builds_sha256=$FRESH_CLASS_BUILDS_SHA256" \
    "classpath_manifest_sha256=$CLASSPATH_MANIFEST_SHA256" \
    "runtime_dependency_manifest_sha256=$RUNTIME_DEPENDENCY_MANIFEST_SHA256" \
    "maven_dependency_arguments_sha256=$MAVEN_DEPENDENCY_ARGUMENTS_SHA256" \
    "object_stage_inputs_sha256=$OBJECT_STAGE_INPUTS_SHA256" \
    "javacpp_reachability_generator_sha256=$JAVACPP_REACHABILITY_GENERATOR_SHA256" \
    "javacpp_reachability_config_sha256=$JAVACPP_REACHABILITY_CONFIG_SHA256" \
    "javacpp_jni_config_sha256=$JAVACPP_JNI_CONFIG_SHA256" \
    "javacpp_initialization_config_sha256=$JAVACPP_INITIALIZATION_CONFIG_SHA256" \
    "javacpp_reachability_manifest_sha256=$JAVACPP_REACHABILITY_MANIFEST_SHA256" \
    "native_image_build_mode=$NATIVE_IMAGE_BUILD_MODE" \
    "native_image_optimization=$NATIVE_IMAGE_OPTIMIZATION" \
    "native_image_optimization_config_sha256=$NATIVE_IMAGE_OPTIMIZATION_CONFIG_SHA256" \
	    "object_sha256=$OBJECT_SHA256" \
	    "base_sdk_sha256=$BASE_SDK_INPUT_SHA256" \
    "base_sdk_receipt_sha256=$BASE_SDK_RECEIPT_SHA256" \
    "base_sdk_native_sha256=$BASE_SDK_NATIVE_SHA256" \
    "process_blas_symbols_capability=$PROCESS_BLAS_SYMBOLS_ABI" \
    "build_script_sha256=$BUILD_SCRIPT_SHA256" \
    "native_image_cache_helper_sha256=$NATIVE_IMAGE_CACHE_HELPER_SHA256" \
	    "object_builder_sha256=$OBJECT_BUILDER_SHA256" \
	    "jdk_support_receipt_sha256=$JDK_SUPPORT_RECEIPT_SHA256" \
	    "libjvm_sha256=$LIBJVM_SHA256" \
	    "liblibchelper_sha256=$LIBLIBCHELPER_SHA256" \
    "maven_sha256=$MAVEN_SHA256" \
    "maven_version_sha256=$MAVEN_VERSION_SHA256" \
    "java_version_sha256=$JAVA_VERSION_SHA256" \
    "linker_script_sha256=$LINKER_SCRIPT_SHA256" \
    "javacpp_jar_sha256=$JAVACPP_JAR_SHA256" \
    "ndk_revision_sha256=$NDK_REVISION_SHA256" \
    "graalvm_version_sha256=$GRAALVM_VERSION_SHA256" \
    "libsdx_sha256=$LIBSDX_SHA256" \
    "libjnijavacpp_sha256=$LIBJNIJAVACPP_SHA256" \
    "jnijavacpp_source_sha256=$JNIJAVACPP_SOURCE_SHA256" \
    "javacpp_lifecycle_source_sha256=$JAVACPP_LIFECYCLE_SOURCE_SHA256" \
    "native_manifest_sha256=$NATIVE_MANIFEST_SHA256" \
    "sdk_native_bytes_sha256=$SDK_NATIVE_BYTES_SHA256" |
    sha256sum | cut -d ' ' -f 1
)"

RECEIPT="$METADATA_DIR/build-receipt"
cat >"$RECEIPT" <<RECEIPT
format=9
stage=android-aot-sdk
inputs_sha256=$INPUTS_SHA256
native_image_build_mode=$NATIVE_IMAGE_BUILD_MODE
native_image_optimization=$NATIVE_IMAGE_OPTIMIZATION
native_image_optimization_config_sha256=$NATIVE_IMAGE_OPTIMIZATION_CONFIG_SHA256
source_manifest_sha256=$SOURCE_MANIFEST_SHA256
classes_sha256=$CLASSES_SHA256
model_classes_sha256=$MODEL_CLASSES_SHA256
fresh_class_builds_sha256=$FRESH_CLASS_BUILDS_SHA256
classpath_manifest_sha256=$CLASSPATH_MANIFEST_SHA256
runtime_dependency_manifest_sha256=$RUNTIME_DEPENDENCY_MANIFEST_SHA256
maven_dependency_arguments_sha256=$MAVEN_DEPENDENCY_ARGUMENTS_SHA256
object_stage_inputs_sha256=$OBJECT_STAGE_INPUTS_SHA256
javacpp_reachability_generator=$JAVACPP_REACHABILITY_GENERATOR
javacpp_reachability_generator_sha256=$JAVACPP_REACHABILITY_GENERATOR_SHA256
javacpp_reachability_config_sha256=$JAVACPP_REACHABILITY_CONFIG_SHA256
javacpp_jni_config_sha256=$JAVACPP_JNI_CONFIG_SHA256
javacpp_initialization_config_sha256=$JAVACPP_INITIALIZATION_CONFIG_SHA256
javacpp_reachability_manifest_sha256=$JAVACPP_REACHABILITY_MANIFEST_SHA256
object_sha256=$OBJECT_SHA256
base_sdk=$BASE_SDK_INPUT
base_sdk_sha256=$BASE_SDK_INPUT_SHA256
base_sdk_receipt_sha256=$BASE_SDK_RECEIPT_SHA256
base_sdk_native_sha256=$BASE_SDK_NATIVE_SHA256
process_blas_symbols_abi=1
process_blas_symbols_capability=$PROCESS_BLAS_SYMBOLS_ABI
build_script=$SCRIPT_DIR/build-android-aot-sdk.sh
build_script_sha256=$BUILD_SCRIPT_SHA256
native_image_cache_helper_sha256=$NATIVE_IMAGE_CACHE_HELPER_SHA256
object_builder_sha256=$OBJECT_BUILDER_SHA256
jdk_support_receipt_sha256=$JDK_SUPPORT_RECEIPT_SHA256
libjvm_sha256=$LIBJVM_SHA256
liblibchelper_sha256=$LIBLIBCHELPER_SHA256
maven=$MAVEN
maven_sha256=$MAVEN_SHA256
maven_version_sha256=$MAVEN_VERSION_SHA256
java_home=$JAVA_HOME_ARG
java_version_sha256=$JAVA_VERSION_SHA256
linker_script_sha256=$LINKER_SCRIPT_SHA256
javacpp_jar=$JAVACPP_JAR
javacpp_jar_sha256=$JAVACPP_JAR_SHA256
ndk_revision_sha256=$NDK_REVISION_SHA256
graalvm_version_sha256=$GRAALVM_VERSION_SHA256
android_api=$ANDROID_API
android_abi=arm64-v8a
libsdx_sha256=$LIBSDX_SHA256
libjnijavacpp_sha256=$LIBJNIJAVACPP_SHA256
jnijavacpp_source_sha256=$JNIJAVACPP_SOURCE_SHA256
javacpp_lifecycle_source_sha256=$JAVACPP_LIFECYCLE_SOURCE_SHA256
native_manifest_sha256=$NATIVE_MANIFEST_SHA256
sdk_native_bytes_sha256=$SDK_NATIVE_BYTES_SHA256
native_library_count=$NATIVE_COUNT
RECEIPT
RECEIPT_SHA256="$(sha256_file "$RECEIPT")"
GENERATION_KEY="$RECEIPT_SHA256"

cat >"$METADATA_DIR/build.properties" <<PROPERTIES
abi.version=2
android.abi=arm64-v8a
android.api=$ANDROID_API
android.ndk=$NDK_REVISION
javacpp.platform=android-arm64
graalvm.java=$EXPECTED_GRAAL_JAVA
native.image=$EXPECTED_NATIVE_IMAGE
native.image.build.mode=$NATIVE_IMAGE_BUILD_MODE
native.image.optimization=-O$NATIVE_IMAGE_OPTIMIZATION
native.image.optimization.config.sha256=$NATIVE_IMAGE_OPTIMIZATION_CONFIG_SHA256
graph.object.sha256=$OBJECT_SHA256
library.sha256=$LIBSDX_SHA256
backend.artifactId=nd4j-native
native.library.count=$NATIVE_COUNT
direct.gguf=true
source.manifest.sha256=$SOURCE_MANIFEST_SHA256
fresh.class.builds.sha256=$FRESH_CLASS_BUILDS_SHA256
javacpp.reachability.sha256=$JAVACPP_REACHABILITY_CONFIG_SHA256
javacpp.jni.sha256=$JAVACPP_JNI_CONFIG_SHA256
javacpp.initialization.sha256=$JAVACPP_INITIALIZATION_CONFIG_SHA256
build.receipt.sha256=$RECEIPT_SHA256
PROPERTIES

ARCHIVE_NAME="sdx-aot-$SDK_VERSION-android-arm64.zip"
ARCHIVE="$STAGE/$ARCHIVE_NAME"
archive_native_members=()
while IFS= read -r library_name; do
  archive_native_members+=("jni/arm64-v8a/$library_name")
done <"$NATIVE_MANIFEST"
(
  cd "$STAGE"
  zip -X -q "$ARCHIVE" \
    include/sdx_llm_c.h \
    metadata/build.properties \
    metadata/build-receipt \
    metadata/jdk-support-receipt \
    metadata/base-sdk-native-bytes.txt \
    metadata/classpath-bytes.txt \
    metadata/fresh-class-builds.txt \
    metadata/javacpp-native-image-reachability.txt \
    metadata/native-image-optimization.txt \
    metadata/cmake-owned-native-libraries.txt \
    metadata/sdk-native-bytes.txt \
    metadata/native-dependency-closure.txt \
    metadata/jnijavacpp.cpp \
    metadata/javacpp_jni_lifecycle.cpp \
    "${archive_native_members[@]}"
)
ARCHIVE_SHA256="$(sha256_file "$ARCHIVE")"
cat >"$STAGE/.complete.cmake" <<COMPLETE
set(SDX_COMPLETED_GENERATION_KEY [[$GENERATION_KEY]])
set(SDX_COMPLETED_LIBRARY_SHA256 [[$LIBSDX_SHA256]])
set(SDX_COMPLETED_ARCHIVE_NAME [[$ARCHIVE_NAME]])
set(SDX_COMPLETED_ARCHIVE_SHA256 [[$ARCHIVE_SHA256]])
COMPLETE

GENERATION_NAME="$GENERATION_KEY-${ARCHIVE_SHA256:0:16}"
GENERATION_DIR="$GENERATIONS_DIR/$GENERATION_NAME"
chmod -R a-w "$STAGE"
# Keep the unpublished staging root owner-writable until the atomic rename.
# Some filesystems reject renaming a read-only directory even when both parent
# directories are writable. Every child remains sealed throughout publication.
chmod u+w "$STAGE"
if [[ -e "$GENERATION_DIR" ]]; then
  [[ -d "$GENERATION_DIR" && ! -L "$GENERATION_DIR" ]] ||
    fail "content-addressed generation path is not a directory: $GENERATION_DIR"
  if find "$GENERATION_DIR" -type l -print -quit | grep -q . ||
     find "$GENERATION_DIR" -perm /0222 -print -quit | grep -q . ||
     ! diff -qr --no-dereference "$STAGE" "$GENERATION_DIR" >/dev/null; then
    fail "existing content-addressed generation differs from rebuilt bytes: $GENERATION_DIR"
  fi
  chmod -R u+w "$STAGE"
  rm -rf -- "$STAGE"
else
  mv -T "$STAGE" "$GENERATION_DIR"
  chmod a-w "$GENERATION_DIR"
fi
find "$GENERATION_DIR" -perm /0222 -print -quit | grep -q . &&
  fail "published content-addressed generation contains writable paths: $GENERATION_DIR"
PUBLIC_TMP="$OUTPUT_PARENT/.android-aot-link.$$"
ln -s ".android-aot-generations/$GENERATION_NAME" "$PUBLIC_TMP"
mv -Tf "$PUBLIC_TMP" "$OUTPUT_LINK"
trap - EXIT
if [[ "$KEEP_WORK" == 1 ]]; then
  printf 'Preserved SDX Android AOT build root: %s\n' "$BUILD_ROOT"
else
  chmod -R u+w "$BUILD_ROOT" 2>/dev/null || true
  rm -rf -- "$BUILD_ROOT"
fi

printf 'Published SDX Android AOT SDK: %s\n' "$OUTPUT_LINK"
printf '  generation: %s\n' "$GENERATION_DIR"
printf '  receipt:    %s\n' "$GENERATION_DIR/metadata/build-receipt"
printf '  receiptSha: %s\n' "$RECEIPT_SHA256"
printf '  libsdxSha:  %s\n' "$LIBSDX_SHA256"
