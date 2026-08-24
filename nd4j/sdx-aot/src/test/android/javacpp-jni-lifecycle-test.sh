#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ANDROID_MAIN_DIR="$SCRIPT_DIR/../../main/android"
LIFECYCLE_SOURCE="$ANDROID_MAIN_DIR/javacpp_jni_lifecycle.cpp"
BUILD_SCRIPT="$ANDROID_MAIN_DIR/build-android-aot-sdk.sh"

fail() {
  printf 'FAIL: %s\n' "$*" >&2
  exit 1
}

[[ -s "$LIFECYCLE_SOURCE" ]] || fail "JavaCPP lifecycle bridge is missing"
[[ -s "$BUILD_SCRIPT" ]] || fail "Android AOT build script is missing"

grep -Fq 'JNI_OnLoad_jnijavacpp(JavaVM* vm, void* reserved)' "$LIFECYCLE_SOURCE" ||
  fail "lifecycle bridge does not declare JavaCPP's generated load hook"
grep -Fq 'JNI_OnUnload_jnijavacpp(JavaVM* vm, void* reserved)' "$LIFECYCLE_SOURCE" ||
  fail "lifecycle bridge does not declare JavaCPP's generated unload hook"
grep -Fq 'JNI_OnLoad(JavaVM* vm, void* reserved)' "$LIFECYCLE_SOURCE" ||
  fail "lifecycle bridge does not export the canonical JNI load hook"
grep -Fq 'return JNI_OnLoad_jnijavacpp(vm, reserved);' "$LIFECYCLE_SOURCE" ||
  fail "canonical JNI load hook does not initialize JavaCPP"
grep -Fq 'JNI_OnUnload(JavaVM* vm, void* reserved)' "$LIFECYCLE_SOURCE" ||
  fail "lifecycle bridge does not export the canonical JNI unload hook"
grep -Fq 'JNI_OnUnload_jnijavacpp(vm, reserved);' "$LIFECYCLE_SOURCE" ||
  fail "canonical JNI unload hook does not release JavaCPP state"

grep -Fq '"$BRIDGE_DIR/jnijavacpp.cpp" "$JAVACPP_LIFECYCLE_BRIDGE"' "$BUILD_SCRIPT" ||
  fail "standalone JavaCPP library is not linked with the lifecycle bridge"
grep -Fq 'for symbol in JNI_OnLoad JNI_OnUnload JNI_OnLoad_jnijavacpp JNI_OnUnload_jnijavacpp' "$BUILD_SCRIPT" ||
  fail "Android build does not verify the complete JavaCPP lifecycle export contract"
if grep -Eq 'ai[_./]kompile|libjnisdx_llm|sdx_llm_android_jni' "$BUILD_SCRIPT" "$ANDROID_MAIN_DIR"/*.cpp; then
  fail "SDX producer contains a consumer-owned Kompile JNI bridge"
fi
if grep -Rq 'KOMPILE_NATIVE_' "$ANDROID_MAIN_DIR"; then
  fail "SDX producer still exposes Kompile-owned cache configuration"
fi

printf 'PASS: standalone JavaCPP lifecycle is verified and consumer JNI stays outside SDX\n'
