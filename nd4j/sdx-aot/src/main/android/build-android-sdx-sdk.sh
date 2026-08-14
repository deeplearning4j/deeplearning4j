#!/usr/bin/env bash
# Auto-configured entrypoint for the independently cached Android SDX SDK stages.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: build-android-sdx-sdk.sh [all|cpu|aot] [options]

The default mode is "all". Local tools are discovered from standard environment
variables, the Android SDK, SDKMAN, Maven local storage, PATH, and the sibling
kompile checkout used by the Android Native Image toolchain.

Common options:
  --android-api N          Android API (default: 28)
  --jobs N                 Parallel jobs (default: min(host CPUs, 8))
  --output-root DIR        Published SDK/cache root (default: $TMPDIR/sdx-android-build)
  --sdk-version VERSION    AOT archive version (default: 1.0.0-SNAPSHOT)
  --offline                Require Maven and object-builder offline mode (default)
  --online                 Permit dependency resolution
  --dev                    Native Image -Ob quick build (default)
  --production             Optimized Native Image production build
  quick-build default      KOMPILE_NATIVE_QUICK_BUILD=1 (0 selects production)
  --keep-work              Preserve a failed AOT generation
  --fresh-classes-only     Compile/audit AOT cache inputs, then stop
  --print-config           Print the resolved configuration and exit
  generation retention     $SDX_ANDROID_GENERATION_RETENTION (default: 2)
  -h, --help               Show this help

Discovery overrides:
  --android-ndk DIR        or SDX_ANDROID_NDK / ANDROID_NDK_HOME
  --java-home DIR          or SDX_JAVA17_HOME / JAVA_HOME
  --maven FILE             or SDX_MAVEN / MAVEN_HOME
  --ccache FILE            or SDX_CCACHE
  --graalvm-home DIR       or SDX_GRAALVM_HOME / GRAALVM_HOME
  --object-builder FILE    or SDX_OBJECT_BUILDER
  sibling support root     SDX_KOMPILE_GRAPH_ROOT
  --javacpp-jar FILE       or SDX_JAVACPP_JAR
  --reuse-jdk-libs DIR     or SDX_JDK_SUPPORT_DIR
  --reuse-svm-libs DIR     or SDX_SVM_SUPPORT_DIR
  --base-sdk DIR           or SDX_CPU_BASE_SDK

Examples:
  build-android-sdx-sdk.sh
  build-android-sdx-sdk.sh cpu
  build-android-sdx-sdk.sh aot
  build-android-sdx-sdk.sh aot --production
USAGE
}

fail() {
  printf 'build-android-sdx-sdk: %s\n' "$*" >&2
  exit 3
}

resolve_executable() {
  local label="$1"
  local explicit="$2"
  shift 2
  local candidate resolved
  if [[ -n "$explicit" ]]; then
    if [[ "$explicit" == */* ]]; then
      [[ -x "$explicit" ]] || fail "$label override is not executable: $explicit"
      realpath -e -- "$explicit"
    else
      resolved="$(command -v -- "$explicit" 2>/dev/null || true)"
      [[ -n "$resolved" ]] || fail "$label override is not on PATH: $explicit"
      realpath -e -- "$resolved"
    fi
    return
  fi
  for candidate in "$@"; do
    [[ -n "$candidate" && -x "$candidate" ]] || continue
    realpath -e -- "$candidate"
    return
  done
  fail "could not discover $label; set its SDX_* override or pass the matching option"
}

resolve_directory() {
  local label="$1"
  local explicit="$2"
  shift 2
  local candidate
  if [[ -n "$explicit" ]]; then
    [[ -d "$explicit" ]] || fail "$label override is not a directory: $explicit"
    realpath -e -- "$explicit"
    return
  fi
  for candidate in "$@"; do
    [[ -n "$candidate" && -d "$candidate" ]] || continue
    realpath -e -- "$candidate"
    return
  done
  fail "could not discover $label; set its SDX_* override or pass the matching option"
}

resolve_regular_file() {
  local label="$1"
  local explicit="$2"
  shift 2
  local candidate
  if [[ -n "$explicit" ]]; then
    [[ -f "$explicit" && ! -L "$explicit" && -s "$explicit" ]] ||
      fail "$label override is not a non-empty regular file: $explicit"
    realpath -e -- "$explicit"
    return
  fi
  for candidate in "$@"; do
    [[ -n "$candidate" && -f "$candidate" && ! -L "$candidate" && -s "$candidate" ]] || continue
    realpath -e -- "$candidate"
    return
  done
  fail "could not discover $label; set its SDX_* override or pass the matching option"
}

java_home_matches_major() {
  local candidate="$1"
  local major="$2"
  local version
  [[ -x "$candidate/bin/java" ]] || return 1
  version="$("$candidate/bin/java" -version 2>&1 | sed -n '1p')"
  [[ "$version" == *"\"$major."* ]]
}

resolve_java_home() {
  local label="$1"
  local major="$2"
  local explicit="$3"
  shift 3
  local candidate
  if [[ -n "$explicit" ]]; then
    [[ -d "$explicit" ]] || fail "$label override is not a directory: $explicit"
    java_home_matches_major "$explicit" "$major" ||
      fail "$label override is not a JDK $major home: $explicit"
    realpath -e -- "$explicit"
    return
  fi
  for candidate in "$@"; do
    [[ -n "$candidate" && -d "$candidate" ]] || continue
    java_home_matches_major "$candidate" "$major" || continue
    realpath -e -- "$candidate"
    return
  done
  fail "could not discover $label (JDK $major); set its SDX_* override or pass the matching option"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODULE_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
DL4J_ROOT="$(cd "$MODULE_DIR/../.." && pwd)"
# Maven reactor selectors are relative to the repository root. Make the
# canonical wrapper independent of whichever project or process invoked it.
cd "$DL4J_ROOT"
CPU_SCRIPT="$SCRIPT_DIR/build-android-cpu-importer-sdk.sh"
AOT_SCRIPT="$SCRIPT_DIR/build-android-aot-sdk.sh"
PRUNE_SCRIPT="$SCRIPT_DIR/prune-android-sdx-build-cache.sh"
[[ -x "$CPU_SCRIPT" && -x "$AOT_SCRIPT" && -x "$PRUNE_SCRIPT" ]] ||
  fail "canonical Android SDK stage scripts are missing or not executable"

MODE=all
if [[ ${1:-} == all || ${1:-} == cpu || ${1:-} == aot ]]; then
  MODE="$1"
  shift
fi

ANDROID_API=28
HOST_JOBS="$(getconf _NPROCESSORS_ONLN 2>/dev/null || printf '8')"
if (( HOST_JOBS > 8 )); then
  JOBS=8
else
  JOBS="$HOST_JOBS"
fi
BUILD_ROOT="${SDX_ANDROID_BUILD_ROOT:-${TMPDIR:-/tmp}/sdx-android-build}"
SDK_VERSION=1.0.0-SNAPSHOT
OFFLINE=1
KEEP_WORK=0
FRESH_CLASSES_ONLY=0
PRINT_CONFIG=0
case "${KOMPILE_NATIVE_QUICK_BUILD:-1}" in
  1|true|TRUE|yes|YES|on|ON) DEFAULT_AOT_BUILD_MODE=dev ;;
  0|false|FALSE|no|NO|off|OFF) DEFAULT_AOT_BUILD_MODE=production ;;
  *) fail "KOMPILE_NATIVE_QUICK_BUILD must be 0/1 or a boolean value" ;;
esac
AOT_BUILD_MODE="${SDX_ANDROID_AOT_BUILD_MODE:-$DEFAULT_AOT_BUILD_MODE}"

ANDROID_NDK_OVERRIDE="${SDX_ANDROID_NDK:-}"
JAVA17_OVERRIDE="${SDX_JAVA17_HOME:-}"
MAVEN_OVERRIDE="${SDX_MAVEN:-}"
CCACHE_OVERRIDE="${SDX_CCACHE:-}"
GRAALVM_OVERRIDE="${SDX_GRAALVM_HOME:-}"
OBJECT_BUILDER_OVERRIDE="${SDX_OBJECT_BUILDER:-}"
JAVACPP_JAR_OVERRIDE="${SDX_JAVACPP_JAR:-}"
JDK_SUPPORT_OVERRIDE="${SDX_JDK_SUPPORT_DIR:-}"
SVM_SUPPORT_OVERRIDE="${SDX_SVM_SUPPORT_DIR:-}"
BASE_SDK_OVERRIDE="${SDX_CPU_BASE_SDK:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --android-api) ANDROID_API="${2:?missing value for --android-api}"; shift 2 ;;
    --jobs) JOBS="${2:?missing value for --jobs}"; shift 2 ;;
    --output-root) BUILD_ROOT="${2:?missing value for --output-root}"; shift 2 ;;
    --sdk-version) SDK_VERSION="${2:?missing value for --sdk-version}"; shift 2 ;;
    --offline) OFFLINE=1; shift ;;
    --online) OFFLINE=0; shift ;;
    --dev) AOT_BUILD_MODE=dev; shift ;;
    --production) AOT_BUILD_MODE=production; shift ;;
    --keep-work) KEEP_WORK=1; shift ;;
    --fresh-classes-only) FRESH_CLASSES_ONLY=1; shift ;;
    --print-config) PRINT_CONFIG=1; shift ;;
    --android-ndk) ANDROID_NDK_OVERRIDE="${2:?missing value for --android-ndk}"; shift 2 ;;
    --java-home) JAVA17_OVERRIDE="${2:?missing value for --java-home}"; shift 2 ;;
    --maven) MAVEN_OVERRIDE="${2:?missing value for --maven}"; shift 2 ;;
    --ccache) CCACHE_OVERRIDE="${2:?missing value for --ccache}"; shift 2 ;;
    --graalvm-home) GRAALVM_OVERRIDE="${2:?missing value for --graalvm-home}"; shift 2 ;;
    --object-builder) OBJECT_BUILDER_OVERRIDE="${2:?missing value for --object-builder}"; shift 2 ;;
    --javacpp-jar) JAVACPP_JAR_OVERRIDE="${2:?missing value for --javacpp-jar}"; shift 2 ;;
    --reuse-jdk-libs) JDK_SUPPORT_OVERRIDE="${2:?missing value for --reuse-jdk-libs}"; shift 2 ;;
    --reuse-svm-libs) SVM_SUPPORT_OVERRIDE="${2:?missing value for --reuse-svm-libs}"; shift 2 ;;
    --base-sdk) BASE_SDK_OVERRIDE="${2:?missing value for --base-sdk}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) fail "unknown argument: $1" ;;
  esac
done

[[ "$ANDROID_API" =~ ^[0-9]+$ ]] || fail "--android-api must be an integer"
[[ "$JOBS" =~ ^[1-9][0-9]*$ ]] || fail "--jobs must be a positive integer"
[[ "$AOT_BUILD_MODE" == dev || "$AOT_BUILD_MODE" == production ]] ||
  fail "SDX_ANDROID_AOT_BUILD_MODE must be dev or production"
[[ "$FRESH_CLASSES_ONLY" == 0 || "$MODE" != cpu ]] ||
  fail "--fresh-classes-only requires 'aot' or 'all' mode"
BUILD_ROOT="$(realpath -m -- "$BUILD_ROOT")"

SDK_ROOT_CANDIDATES=()
for sdk_root in "${ANDROID_NDK_HOME:-}" "${ANDROID_NDK_ROOT:-}"; do
  [[ -n "$sdk_root" ]] || continue
  SDK_ROOT_CANDIDATES+=("$sdk_root")
done
for sdk_root in "${ANDROID_SDK_ROOT:-}" "${ANDROID_HOME:-}" \
  "${HOME:-}/Android/Sdk" "${HOME:-}/Library/Android/sdk" "${HOME:-}/dev-apps/android-sdk"; do
  [[ -n "$sdk_root" ]] || continue
  SDK_ROOT_CANDIDATES+=("$sdk_root/ndk/28.1.13356709")
done
ANDROID_NDK="$(resolve_directory "Android NDK 28.1.13356709" "$ANDROID_NDK_OVERRIDE" "${SDK_ROOT_CANDIDATES[@]}")"

SDKMAN_JAVA_ROOT="${SDKMAN_CANDIDATES_DIR:-${HOME:-}/.sdkman/candidates}/java"
shopt -s nullglob
JAVA17_CANDIDATES=(
  "${JAVA_HOME:-}"
  "$SDKMAN_JAVA_ROOT/current"
  "$SDKMAN_JAVA_ROOT"/17*
  "${HOME:-}/dev-apps/jdk-17"
)
GRAAL_CANDIDATES=(
  "${GRAALVM_HOME:-}"
  "$SDKMAN_JAVA_ROOT"/21*graal*
  "$SDKMAN_JAVA_ROOT"/graal*
)
shopt -u nullglob
JAVA17_HOME="$(resolve_java_home "managed-build JDK" 17 "$JAVA17_OVERRIDE" "${JAVA17_CANDIDATES[@]}")"

MAVEN_CANDIDATES=()
if [[ -n "${MAVEN_HOME:-}" ]]; then
  MAVEN_CANDIDATES+=("$MAVEN_HOME/bin/mvn")
fi
MAVEN_CANDIDATES+=(
  "$DL4J_ROOT/mvnw"
  "${HOME:-}/dev-apps/mvn/bin/mvn"
  "$(command -v mvn 2>/dev/null || true)"
)
MAVEN="$(resolve_executable "Maven" "$MAVEN_OVERRIDE" "${MAVEN_CANDIDATES[@]}")"

CCACHE=""
if [[ "$MODE" != aot ]]; then
  CCACHE="$(resolve_executable "ccache" "$CCACHE_OVERRIDE"     "$(command -v ccache 2>/dev/null || true)"     "${CONDA_PREFIX:-}/bin/ccache"     "${HOME:-}/miniconda3/bin/ccache")"
fi

KOMPILE_GRAPH_ROOT="${SDX_KOMPILE_GRAPH_ROOT:-$DL4J_ROOT/../kompile/kompile-app/kompile-data/kompile-graphs/kompile-graph-reasoning-local}"
SUPPORT_DEFAULT="$KOMPILE_GRAPH_ROOT/target/android-ndk-aot/clibraries/bionic"
M2_REPOSITORY="${M2_REPOSITORY:-${HOME:-}/.m2/repository}"

GRAALVM_HOME_RESOLVED=""
OBJECT_BUILDER=""
JAVACPP_JAR=""
JDK_SUPPORT_DIR=""
SVM_SUPPORT_DIR=""
if [[ "$MODE" != cpu ]]; then
  GRAALVM_HOME_RESOLVED="$(resolve_java_home "GraalVM" 21 "$GRAALVM_OVERRIDE" "${GRAAL_CANDIDATES[@]}")"
  OBJECT_BUILDER="$(resolve_executable "Android Native Image object builder" "$OBJECT_BUILDER_OVERRIDE"     "$KOMPILE_GRAPH_ROOT/build-android-ndk.sh")"
  JAVACPP_JAR="$(resolve_regular_file "JavaCPP builder jar" "$JAVACPP_JAR_OVERRIDE"     "$M2_REPOSITORY/org/bytedeco/javacpp/1.5.13/javacpp-1.5.13.jar")"
  JDK_SUPPORT_DIR="$(resolve_directory "Android JDK support closure" "$JDK_SUPPORT_OVERRIDE" "$SUPPORT_DEFAULT")"
  SVM_SUPPORT_DIR="$(resolve_directory "Android SVM support closure" "$SVM_SUPPORT_OVERRIDE" "$SUPPORT_DEFAULT")"
fi

CPU_LINK="$BUILD_ROOT/cpu-sdk/current"
CPU_WORK="$BUILD_ROOT/cpu-sdk/work"
AOT_LINK="$BUILD_ROOT/aot-sdk/current"
AOT_WORK="$BUILD_ROOT/aot-sdk/work"
BASE_SDK="${BASE_SDK_OVERRIDE:-$CPU_LINK}"

printf 'Resolved Android SDX build configuration:\n'
printf '  mode:           %s\n' "$MODE"
printf '  Android NDK:    %s\n' "$ANDROID_NDK"
printf '  managed JDK:    %s\n' "$JAVA17_HOME"
printf '  Maven:          %s\n' "$MAVEN"
printf '  build root:     %s\n' "$BUILD_ROOT"
printf '  offline:        %s\n' "$OFFLINE"
printf '  jobs:           %s\n' "$JOBS"
if [[ "$MODE" != aot ]]; then
  printf '  ccache:         %s\n' "$CCACHE"
  printf '  CPU SDK:        %s\n' "$CPU_LINK"
fi
if [[ "$MODE" != cpu ]]; then
  printf '  AOT build mode: %s\n' "$AOT_BUILD_MODE"
  printf '  GraalVM:        %s\n' "$GRAALVM_HOME_RESOLVED"
  printf '  object builder: %s\n' "$OBJECT_BUILDER"
  printf '  JavaCPP:        %s\n' "$JAVACPP_JAR"
  printf '  JDK support:    %s\n' "$JDK_SUPPORT_DIR"
  printf '  SVM support:    %s\n' "$SVM_SUPPORT_DIR"
  printf '  base SDK:       %s\n' "$BASE_SDK"
  printf '  AOT SDK:        %s\n' "$AOT_LINK"
fi
[[ "$PRINT_CONFIG" == 0 ]] || exit 0

case "${SDX_ANDROID_PIPELINE_LOCK_HELD:-0}" in
  0)
    command -v flock >/dev/null 2>&1 || fail "flock is required"
    mkdir -p -- "$BUILD_ROOT/.locks"
    [[ -d "$BUILD_ROOT/.locks" && ! -L "$BUILD_ROOT/.locks" ]] ||
      fail "pipeline lock root must be a real directory: $BUILD_ROOT/.locks"
    exec {SDK_PIPELINE_LOCK_FD}>"$BUILD_ROOT/.locks/tensor-g3-offline-apk.lock"
    printf 'Waiting for the Android SDK build lock: %s\n' "$BUILD_ROOT/.locks/tensor-g3-offline-apk.lock"
    flock "$SDK_PIPELINE_LOCK_FD"
    export SDX_ANDROID_PIPELINE_LOCK_HELD=1
    ;;
  1) ;;
  *) fail "SDX_ANDROID_PIPELINE_LOCK_HELD must be 0 or 1" ;;
esac

"$PRUNE_SCRIPT" --build-root "$BUILD_ROOT"

offline_args=()
[[ "$OFFLINE" == 0 ]] || offline_args+=(--offline)

if [[ "$MODE" != aot ]]; then
  "$CPU_SCRIPT"     --android-ndk "$ANDROID_NDK"     --java-home "$JAVA17_HOME"     --maven "$MAVEN"     --ccache "$CCACHE"     --android-api "$ANDROID_API"     --jobs "$JOBS"     --output-link "$CPU_LINK"     --work-dir "$CPU_WORK"     "${offline_args[@]}"
  "$PRUNE_SCRIPT" --build-root "$BUILD_ROOT"
fi

if [[ "$MODE" != cpu ]]; then
  [[ -d "$BASE_SDK" || -L "$BASE_SDK" ]] ||
    fail "CPU base SDK is missing: $BASE_SDK (run the default 'all' mode or the 'cpu' mode first)"
  aot_args=(
    --android-ndk "$ANDROID_NDK"
    --graalvm-home "$GRAALVM_HOME_RESOLVED"
    --object-builder "$OBJECT_BUILDER"
    --maven "$MAVEN"
    --java-home "$JAVA17_HOME"
    --javacpp-jar "$JAVACPP_JAR"
    --base-sdk "$BASE_SDK"
    --reuse-jdk-libs "$JDK_SUPPORT_DIR"
    --reuse-svm-libs "$SVM_SUPPORT_DIR"
    --work-dir "$AOT_WORK"
    --output-link "$AOT_LINK"
    --sdk-version "$SDK_VERSION"
    --jobs "$JOBS"
    "${offline_args[@]}"
  )
  if [[ "$AOT_BUILD_MODE" == dev ]]; then
    aot_args+=(--quick-build)
  else
    aot_args+=(--production)
  fi
  [[ "$KEEP_WORK" == 0 ]] || aot_args+=(--keep-work)
  [[ "$FRESH_CLASSES_ONLY" == 0 ]] || aot_args+=(--fresh-classes-only)
  "$AOT_SCRIPT" "${aot_args[@]}"
  "$PRUNE_SCRIPT" --build-root "$BUILD_ROOT"
fi
