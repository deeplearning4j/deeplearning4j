#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../../main/android/native-image-object-identity.sh
source "$SCRIPT_DIR/../../main/android/native-image-object-identity.sh"

fail() {
  printf 'FAIL: %s\n' "$*" >&2
  exit 1
}

WORK_DIR="$(mktemp -d)"
trap 'rm -rf -- "$WORK_DIR"' EXIT

make_archive() {
  local destination="${1:?destination is required}"
  local native_payload="${2:?native payload is required}"
  local managed_payload="${3:?managed payload is required}"
  local native_name="${4:-libnative.so}"
  local root="$WORK_DIR/root"

  rm -rf -- "$root"
  mkdir -p -- "$root/pkg" "$root/META-INF/services" "$root/META-INF/native/android-arm64"
  printf '%s\n' "$managed_payload" >"$root/pkg/Controlled.class"
  printf '%s\n' 'pkg.Controlled' >"$root/META-INF/services/example.Service"
  printf '%s\n' "$native_payload" >"$root/META-INF/native/android-arm64/$native_name"
  (
    cd "$root"
    zip -q -X -r "$destination" .
  )
}

archive_a="$WORK_DIR/a.jar"
archive_native_changed="$WORK_DIR/native-changed.jar"
archive_managed_changed="$WORK_DIR/managed-changed.jar"
archive_native_name_changed="$WORK_DIR/native-name-changed.jar"
make_archive "$archive_a" native-v1 managed-v1
make_archive "$archive_native_changed" native-v2 managed-v1
make_archive "$archive_managed_changed" native-v1 managed-v2
make_archive "$archive_native_name_changed" native-v1 managed-v1 librenamed.so

hash_a="$(sdx_native_image_jar_analysis_sha256 "$archive_a")"
hash_native_changed="$(sdx_native_image_jar_analysis_sha256 "$archive_native_changed")"
hash_managed_changed="$(sdx_native_image_jar_analysis_sha256 "$archive_managed_changed")"
hash_native_name_changed="$(sdx_native_image_jar_analysis_sha256 "$archive_native_name_changed")"

[[ "$hash_a" == "$hash_native_changed" ]] ||
  fail "native payload bytes invalidated managed analysis identity"
[[ "$hash_a" != "$hash_managed_changed" ]] ||
  fail "managed bytecode did not invalidate analysis identity"
[[ "$hash_a" != "$hash_native_name_changed" ]] ||
  fail "native closure entry-name change did not invalidate analysis identity"

SOURCE_MANIFEST_SHA256=source
RUNTIME_ANALYSIS_MANIFEST_SHA256=runtime
MAVEN_DEPENDENCY_ARGUMENTS_SHA256=maven-args
JAVACPP_REACHABILITY_GENERATOR_SHA256=generator
JAVACPP_REACHABILITY_CONFIG_SHA256=reachability
JAVACPP_JNI_CONFIG_SHA256=jni
JAVACPP_INITIALIZATION_CONFIG_SHA256=initialization
JAVACPP_REACHABILITY_MANIFEST_SHA256=manifest
NATIVE_IMAGE_BUILD_MODE=dev
NATIVE_IMAGE_OPTIMIZATION=b
NATIVE_IMAGE_OPTIMIZATION_CONFIG_SHA256=quick
NATIVE_IMAGE_OBJECT_IDENTITY_HELPER_SHA256=identity-helper
OBJECT_BUILDER_SHA256=builder
LIBJVM_SHA256=svm-jvm
LIBLIBCHELPER_SHA256=svm-helper
MAVEN_SHA256=maven
MAVEN_VERSION_SHA256=maven-version
JAVA_VERSION_SHA256=java-version
JAVACPP_JAR_SHA256=javacpp
NDK_REVISION_SHA256=ndk
GRAALVM_VERSION_SHA256=graal
ANDROID_API=28

identity_a="$(sdx_native_image_object_identity_sha256)"
JAVACPP_JNI_CONFIG_SHA256=jni-changed
identity_jni_changed="$(sdx_native_image_object_identity_sha256)"
[[ "$identity_a" != "$identity_jni_changed" ]] ||
  fail "JNI reachability metadata did not invalidate the Graal object identity"
JAVACPP_JNI_CONFIG_SHA256=jni

BASE_SDK_INPUT_SHA256=base-sdk-changed
BASE_SDK_RECEIPT_SHA256=base-receipt-changed
BASE_SDK_NATIVE_SHA256=base-native-changed
JDK_SUPPORT_RECEIPT_SHA256=jdk-link-input-changed
BUILD_SCRIPT_SHA256=wrapper-changed
NATIVE_IMAGE_CACHE_HELPER_SHA256=cache-wrapper-changed
identity_native_only_changed="$(sdx_native_image_object_identity_sha256)"
[[ "$identity_a" == "$identity_native_only_changed" ]] ||
  fail "native/link/wrapper inputs invalidated the Graal object identity"

RUNTIME_ANALYSIS_MANIFEST_SHA256=managed-runtime-changed
identity_managed_changed="$(sdx_native_image_object_identity_sha256)"
[[ "$identity_a" != "$identity_managed_changed" ]] ||
  fail "managed runtime change did not invalidate the Graal object identity"

printf 'PASS: native payload changes reuse the Graal object; managed changes invalidate it\n'
