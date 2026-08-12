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
  --jobs N                   Native support build jobs
  -h, --help                 Show this help
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

dl4j_aot_source_manifest_sha256() {
  local relative file mode digest
  local -a roots=(
    nd4j/sdx-aot
    nd4j/nd4j-backends/nd4j-backend-impls/nd4j-sdx
    nd4j/nd4j-backends/nd4j-backend-impls/nd4j-sdx-model
    nd4j/nd4j-backends/nd4j-backend-impls/nd4j-sdx-preset
  )
  {
    git -C "$DL4J_ROOT" ls-files -z --cached --others --exclude-standard -- "${roots[@]}" |
      LC_ALL=C sort -z |
      while IFS= read -r -d '' relative; do
        file="$DL4J_ROOT/$relative"
        [[ -f "$file" ]] || continue
        mode="$(stat -c '%a' "$file")"
        digest="$(sha256_file "$file")"
        printf '%s\0%s\0%s\0' "$relative" "$mode" "$digest"
      done
  } | sha256sum | cut -d ' ' -f 1
}

module_source_manifest_sha256() {
  local root="$1"
  local relative file mode digest
  {
    git -C "$DL4J_ROOT" ls-files -z --cached --others --exclude-standard -- "$root" |
      LC_ALL=C sort -z |
      while IFS= read -r -d '' relative; do
        file="$DL4J_ROOT/$relative"
        [[ -f "$file" ]] || continue
        mode="$(stat -c '%a' "$file")"
        digest="$(sha256_file "$file")"
        printf '%s\0%s\0%s\0' "$relative" "$mode" "$digest"
      done
  } | sha256sum | cut -d ' ' -f 1
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
  "producer_sha256=$OBJECT_BUILDER_SHA256" \
  "android_ndk_revision=$NDK_REVISION" \
  "android_ndk_source_properties_sha256=$(sha256_file "$ANDROID_NDK/source.properties")" \
  "graalvm_version_sha256=$(printf '%s' "$NATIVE_IMAGE_VERSION" | sha256sum | cut -d ' ' -f 1)"; do
  [[ "$(grep -Fxc "$expected" "$JDK_SUPPORT_RECEIPT")" -eq 1 ]] ||
    fail "JDK support receipt is missing or duplicates: $expected"
done
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

OUTPUT_PARENT="$(dirname "$OUTPUT_LINK")"
GENERATIONS_DIR="$OUTPUT_PARENT/.android-aot-generations"
mkdir -p "$WORK_DIR" "$GENERATIONS_DIR"
BUILD_ROOT="$(mktemp -d "$WORK_DIR/generation.XXXXXXXX")"
trap 'chmod -R u+w "$BUILD_ROOT" 2>/dev/null || true; rm -rf -- "$BUILD_ROOT"' EXIT
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

SOURCE_MANIFEST_SHA256="$(dl4j_aot_source_manifest_sha256)"
FRESH_CLASSES_ROOT="$BUILD_ROOT/fresh-classes"
mkdir -p "$FRESH_CLASSES_ROOT"
declare -A MODULE_ROOTS=(
  [nd4j-sdx-preset]="nd4j/nd4j-backends/nd4j-backend-impls/nd4j-sdx-preset"
  [nd4j-sdx-model]="nd4j/nd4j-backends/nd4j-backend-impls/nd4j-sdx-model"
  [nd4j-sdx]="nd4j/nd4j-backends/nd4j-backend-impls/nd4j-sdx"
  [sdx-aot]="nd4j/sdx-aot"
)
FRESH_COMPILE_ORDER=(nd4j-sdx-preset nd4j-sdx-model nd4j-sdx sdx-aot)
maven_compile_flags=(-DskipTests)
[[ "$OFFLINE" == 0 ]] || maven_compile_flags+=(-o)
for module_id in "${FRESH_COMPILE_ORDER[@]}"; do
  module_root="$DL4J_ROOT/${MODULE_ROOTS[$module_id]}"
  module_classes="$FRESH_CLASSES_ROOT/$module_id"
  mkdir -p "$module_classes"
  env JAVA_HOME="$JAVA_HOME_ARG" PATH="$JAVA_HOME_ARG/bin:$PATH" \
    "$MAVEN" "${maven_compile_flags[@]}" -f "$module_root/pom.xml" \
      -Dmaven.compiler.outputDirectory="$module_classes" compiler:compile
  [[ -n "$(find "$module_classes" -type f -name '*.class' -print -quit)" ]] ||
    fail "fresh Maven compilation produced no classes for $module_id"
done
AOT_RESOURCES="$MODULE_DIR/src/main/resources"
if [[ -d "$AOT_RESOURCES" ]]; then
  if find "$AOT_RESOURCES" -type l -print -quit | grep -q .; then
    fail "sdx-aot resources contain a symlink"
  fi
  cp -a "$AOT_RESOURCES"/. "$FRESH_CLASSES_ROOT/sdx-aot"/
fi

CLASSES_DIR="$FRESH_CLASSES_ROOT/sdx-aot"
MODEL_CLASSES_DIR="$FRESH_CLASSES_ROOT/nd4j-sdx-model"
declare -A CURRENT_CLASSPATH_DIRS=(
  [nd4j-sdx]="$FRESH_CLASSES_ROOT/nd4j-sdx"
  [nd4j-sdx-model]="$FRESH_CLASSES_ROOT/nd4j-sdx-model"
  [nd4j-sdx-preset]="$FRESH_CLASSES_ROOT/nd4j-sdx-preset"
)
CURRENT_CLASSPATH_IDS=(nd4j-sdx nd4j-sdx-model nd4j-sdx-preset)
[[ -s "$MODEL_CLASSES_DIR/org/nd4j/dsp/model/SdxLlmNative.class" ]] ||
  fail "fresh compilation omitted SdxLlmNative.class"
CLASSPATH_FILE="$BUILD_ROOT/resolved-classpath.txt"
env JAVA_HOME="$JAVA_HOME_ARG" PATH="$JAVA_HOME_ARG/bin:$PATH" \
  "$MAVEN" "${maven_compile_flags[@]}" -f "$MODULE_DIR/pom.xml" \
    dependency:build-classpath -Dmdep.includeScope=runtime -Dmdep.outputFile="$CLASSPATH_FILE"
[[ -s "$CLASSPATH_FILE" ]] || fail "Maven did not resolve the sdx-aot runtime classpath"

CLASSES_SHA256="$(tree_manifest_sha256 "$CLASSES_DIR")"
MODEL_CLASSES_SHA256="$(tree_manifest_sha256 "$MODEL_CLASSES_DIR")"

FRESH_CLASS_BUILDS="$METADATA_DIR/fresh-class-builds.txt"
for module_id in "${FRESH_COMPILE_ORDER[@]}"; do
  printf '%s %s %s\n' \
    "$module_id" \
    "$(module_source_manifest_sha256 "${MODULE_ROOTS[$module_id]}")" \
    "$(tree_manifest_sha256 "$FRESH_CLASSES_ROOT/$module_id")"
done >"$FRESH_CLASS_BUILDS"
FRESH_CLASS_BUILDS_SHA256="$(sha256_file "$FRESH_CLASS_BUILDS")"

EFFECTIVE_CLASSPATH_FILE="$BUILD_ROOT/effective-classpath.txt"
effective_classpath_entries=()
for current_id in "${CURRENT_CLASSPATH_IDS[@]}"; do
  effective_classpath_entries+=("${CURRENT_CLASSPATH_DIRS[$current_id]}")
done
while IFS= read -r classpath_entry; do
  [[ -n "$classpath_entry" ]] || continue
  skip_classpath_entry=0
  classpath_basename="$(basename -- "$classpath_entry")"
  for current_id in "${CURRENT_CLASSPATH_IDS[@]}"; do
    if [[ "$classpath_entry" == "${CURRENT_CLASSPATH_DIRS[$current_id]}" ||
          "$classpath_basename" == "$current_id-"*.jar ]]; then
      skip_classpath_entry=1
      break
    fi
  done
  [[ "$skip_classpath_entry" == "1" ]] || effective_classpath_entries+=("$classpath_entry")
done < <(tr ':' '\n' <"$CLASSPATH_FILE")
(
  IFS=:
  printf '%s\n' "${effective_classpath_entries[*]}"
) >"$EFFECTIVE_CLASSPATH_FILE"

CLASSPATH_MANIFEST="$METADATA_DIR/classpath-bytes.txt"
classpath_index=0
while IFS= read -r classpath_entry; do
  [[ -n "$classpath_entry" ]] || continue
  classpath_index=$((classpath_index + 1))
  if [[ -f "$classpath_entry" ]]; then
    printf '%06d file %s %s\n' "$classpath_index" "$(sha256_file "$classpath_entry")" "$classpath_entry"
  elif [[ -d "$classpath_entry" ]]; then
    printf '%06d tree %s %s\n' "$classpath_index" "$(tree_manifest_sha256 "$classpath_entry")" "$classpath_entry"
  else
    fail "classpath entry is missing: $classpath_entry"
  fi
done < <(tr ':' '\n' <"$EFFECTIVE_CLASSPATH_FILE") >"$CLASSPATH_MANIFEST"
[[ "$classpath_index" -gt 0 ]] || fail "classpath is empty"
CLASSPATH_MANIFEST_SHA256="$(sha256_file "$CLASSPATH_MANIFEST")"

object_args=(
  --android-ndk "$ANDROID_NDK"
  --graalvm-home "$GRAALVM_HOME"
  --jobs "$JOBS"
  --work-dir "$OBJECT_WORK"
  --output-dir "$BUILD_ROOT/unused-graph-sdk"
  --classes-dir "$CLASSES_DIR"
  --classpath-file "$EFFECTIVE_CLASSPATH_FILE"
  --reuse-jdk-libs "$REUSE_JDK_LIBS"
  --reuse-svm-libs "$REUSE_SVM_LIBS"
  --object-output "$OBJECT"
)
[[ "$OFFLINE" == "0" ]] || object_args+=(--offline)
"$OBJECT_BUILDER" "${object_args[@]}"
[[ -s "$OBJECT" ]] || fail "Native Image relocatable object was not produced"
OBJECT_ELF_HEADER="$BUILD_ROOT/libsdx_llm.o.elf-header"
"$LLVM_READELF" -h "$OBJECT" >"$OBJECT_ELF_HEADER"
grep -q 'Machine:.*AArch64' "$OBJECT_ELF_HEADER" ||
  fail "Native Image object is not AArch64"
OBJECT_SHA256="$(sha256_file "$OBJECT")"

UNSTRIPPED="$BUILD_ROOT/libsdx_llm.unstripped.so"
LINKER_SCRIPT="$MODULE_DIR/src/main/linker/sdx_exports.lds"
[[ -s "$LINKER_SCRIPT" ]] || fail "SDX export version script is missing"
"$CLANG" -shared -o "$UNSTRIPPED" "$OBJECT"   -Wl,--start-group     "$REUSE_SVM_LIBS/libjvm.a" "$REUSE_SVM_LIBS/liblibchelper.a"     "$REUSE_JDK_LIBS/libjava.a" "$REUSE_JDK_LIBS/libnet.a"     "$REUSE_JDK_LIBS/libnio.a" "$REUSE_JDK_LIBS/libzip.a"     "$REUSE_JDK_LIBS/libprefs.a" "$REUSE_JDK_LIBS/libextnet.a"   -Wl,--end-group   -ldl -lz -lm -llog   -Wl,--no-undefined -Wl,--gc-sections -Wl,--build-id=sha1   -Wl,-z,relro,-z,now -Wl,-z,max-page-size=16384 -Wl,-z,common-page-size=16384   -Wl,-soname,libsdx_llm.so -Wl,--version-script="$LINKER_SCRIPT"
cp "$UNSTRIPPED" "$JNI_DIR/libsdx_llm.so"
"$LLVM_STRIP" --strip-unneeded "$JNI_DIR/libsdx_llm.so"
assert_unix_file_attributes_abi "$JNI_DIR/libsdx_llm.so"
LIBSDX_DYNAMIC_SYMBOLS="$BUILD_ROOT/libsdx_llm.dynamic-symbols"
"$LLVM_NM" -D --defined-only "$JNI_DIR/libsdx_llm.so" >"$LIBSDX_DYNAMIC_SYMBOLS"
for symbol in sdxLlmAbiVersion sdxLlmPrepareGguf sdxLlmResolveModelBundle sdxLlmLoadCompiledModel sdxLlmGenerateStreaming sdxLlmParseChatResult; do
  grep -q "[[:space:]]${symbol}@@SDX_LLM_1$" "$LIBSDX_DYNAMIC_SYMBOLS" ||
    fail "fresh libsdx_llm.so omitted required versioned export: ${symbol}@@SDX_LLM_1"
done

"$GRAALVM_HOME/bin/java" -cp "$JAVACPP_JAR" org.bytedeco.javacpp.tools.Builder   -classpath "$MODEL_CLASSES_DIR" -d "$BRIDGE_DIR" -o jnisdx_llm -nocompile   org.nd4j.dsp.model.SdxLlmNative
[[ -s "$BRIDGE_DIR/jnijavacpp.cpp" && -s "$BRIDGE_DIR/jnisdx_llm.cpp" ]] ||
  fail "JavaCPP did not generate both Android bridge sources"

COMMON_BRIDGE_FLAGS=(
  -shared -fPIC -O2 -std=c++17 -DANDROID -D__ANDROID_API__="$ANDROID_API"
  -Wl,--no-undefined -Wl,--build-id=sha1
  -Wl,-z,relro,-z,now -Wl,-z,max-page-size=16384 -Wl,-z,common-page-size=16384
)
"$CLANGXX" "${COMMON_BRIDGE_FLAGS[@]}"   -Wl,-soname,libjnijavacpp.so   "$BRIDGE_DIR/jnijavacpp.cpp" -o "$JNI_DIR/libjnijavacpp.so"   -llog -ldl -lm
"$CLANGXX" "${COMMON_BRIDGE_FLAGS[@]}"   -Wl,-soname,libjnisdx_llm.so   -I"$MODULE_DIR/include" "$BRIDGE_DIR/jnisdx_llm.cpp"   "$JNI_DIR/libsdx_llm.so" "$JNI_DIR/libjnijavacpp.so"   -o "$JNI_DIR/libjnisdx_llm.so" -llog -ldl -lm
LIBJNISDX_DYNAMIC_SYMBOLS="$BUILD_ROOT/libjnisdx_llm.dynamic-symbols"
"$LLVM_NM" -D --defined-only "$JNI_DIR/libjnisdx_llm.so" >"$LIBJNISDX_DYNAMIC_SYMBOLS"
grep -q 'sdxLlmParseChatResult' "$LIBJNISDX_DYNAMIC_SYMBOLS" ||
  fail "fresh libjnisdx_llm.so omitted the parse-chat JNI binding"

BASE_NATIVE_BYTES="$METADATA_DIR/base-sdk-native-bytes.txt"
while IFS= read -r library_name; do
  [[ "$library_name" =~ ^lib[A-Za-z0-9._+-]+[.]so$ ]] ||
    fail "unsafe base SDK native member: $library_name"
  case "$library_name" in
    libsdx_llm.so|libjnisdx_llm.so|libjnijavacpp.so|libc++_shared.so) continue ;;
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
cp "$BRIDGE_DIR/jnisdx_llm.cpp" "$METADATA_DIR/jnisdx_llm.cpp"

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
MAVEN_SHA256="$(sha256_file "$MAVEN")"
MAVEN_VERSION="$({ env JAVA_HOME="$JAVA_HOME_ARG" PATH="$JAVA_HOME_ARG/bin:$PATH" "$MAVEN" --version; } 2>&1)"
MAVEN_VERSION_SHA256="$(printf '%s' "$MAVEN_VERSION" | sha256sum | cut -d ' ' -f 1)"
JAVA_VERSION="$({ "$JAVA_HOME_ARG/bin/java" -version; } 2>&1)"
JAVA_VERSION_SHA256="$(printf '%s' "$JAVA_VERSION" | sha256sum | cut -d ' ' -f 1)"
BASE_SDK_NATIVE_SHA256="$(sha256_file "$BASE_NATIVE_BYTES")"
LINKER_SCRIPT_SHA256="$(sha256_file "$LINKER_SCRIPT")"
JAVACPP_JAR_SHA256="$(sha256_file "$JAVACPP_JAR")"
NDK_REVISION_SHA256="$(sha256_file "$ANDROID_NDK/source.properties")"
GRAALVM_VERSION_SHA256="$(printf '%s' "$NATIVE_IMAGE_VERSION" | sha256sum | cut -d ' ' -f 1)"
LIBSDX_SHA256="$(sha256_file "$JNI_DIR/libsdx_llm.so")"
LIBJNISDX_SHA256="$(sha256_file "$JNI_DIR/libjnisdx_llm.so")"
LIBJNIJAVACPP_SHA256="$(sha256_file "$JNI_DIR/libjnijavacpp.so")"
JNIJAVACPP_SOURCE_SHA256="$(sha256_file "$METADATA_DIR/jnijavacpp.cpp")"
JNISDX_SOURCE_SHA256="$(sha256_file "$METADATA_DIR/jnisdx_llm.cpp")"
NATIVE_MANIFEST_SHA256="$(sha256_file "$NATIVE_MANIFEST")"

[[ "$(dl4j_aot_source_manifest_sha256)" == "$SOURCE_MANIFEST_SHA256" ]] ||
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
	    "object_sha256=$OBJECT_SHA256" \
	    "base_sdk_sha256=$BASE_SDK_INPUT_SHA256" \
	    "base_sdk_native_sha256=$BASE_SDK_NATIVE_SHA256" \
    "build_script_sha256=$BUILD_SCRIPT_SHA256" \
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
    "libjnisdx_sha256=$LIBJNISDX_SHA256" \
    "libjnijavacpp_sha256=$LIBJNIJAVACPP_SHA256" \
    "jnijavacpp_source_sha256=$JNIJAVACPP_SOURCE_SHA256" \
    "jnisdx_source_sha256=$JNISDX_SOURCE_SHA256" \
    "native_manifest_sha256=$NATIVE_MANIFEST_SHA256" \
    "sdk_native_bytes_sha256=$SDK_NATIVE_BYTES_SHA256" |
    sha256sum | cut -d ' ' -f 1
)"

RECEIPT="$METADATA_DIR/build-receipt"
cat >"$RECEIPT" <<RECEIPT
format=3
stage=android-aot-sdk
inputs_sha256=$INPUTS_SHA256
source_manifest_sha256=$SOURCE_MANIFEST_SHA256
classes_sha256=$CLASSES_SHA256
model_classes_sha256=$MODEL_CLASSES_SHA256
fresh_class_builds_sha256=$FRESH_CLASS_BUILDS_SHA256
classpath_manifest_sha256=$CLASSPATH_MANIFEST_SHA256
object_sha256=$OBJECT_SHA256
base_sdk=$BASE_SDK_INPUT
base_sdk_sha256=$BASE_SDK_INPUT_SHA256
base_sdk_native_sha256=$BASE_SDK_NATIVE_SHA256
build_script=$SCRIPT_DIR/build-android-aot-sdk.sh
build_script_sha256=$BUILD_SCRIPT_SHA256
object_builder=$OBJECT_BUILDER
object_builder_sha256=$OBJECT_BUILDER_SHA256
jdk_support_receipt_sha256=$JDK_SUPPORT_RECEIPT_SHA256
svm_support=$REUSE_SVM_LIBS_SOURCE
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
libjnisdx_sha256=$LIBJNISDX_SHA256
libjnijavacpp_sha256=$LIBJNIJAVACPP_SHA256
jnijavacpp_source_sha256=$JNIJAVACPP_SOURCE_SHA256
jnisdx_source_sha256=$JNISDX_SOURCE_SHA256
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
graph.object.sha256=$OBJECT_SHA256
library.sha256=$LIBSDX_SHA256
backend.artifactId=nd4j-native
native.library.count=$NATIVE_COUNT
direct.gguf=true
source.manifest.sha256=$SOURCE_MANIFEST_SHA256
fresh.class.builds.sha256=$FRESH_CLASS_BUILDS_SHA256
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
    metadata/cmake-owned-native-libraries.txt \
    metadata/sdk-native-bytes.txt \
    metadata/native-dependency-closure.txt \
    metadata/jnijavacpp.cpp \
    metadata/jnisdx_llm.cpp \
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
rm -rf -- "$BUILD_ROOT"

printf 'Published SDX Android AOT SDK: %s\n' "$OUTPUT_LINK"
printf '  generation: %s\n' "$GENERATION_DIR"
printf '  receipt:    %s\n' "$GENERATION_DIR/metadata/build-receipt"
printf '  receiptSha: %s\n' "$RECEIPT_SHA256"
printf '  libsdxSha:  %s\n' "$LIBSDX_SHA256"
