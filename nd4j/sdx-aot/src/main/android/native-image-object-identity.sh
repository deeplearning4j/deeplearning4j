#!/usr/bin/env bash

# Stable identity for the completed Android GraalVM relocatable object.
#
# This is deliberately narrower than the final SDK identity. Native libraries
# from libnd4j, OpenBLAS, tokenizers, providers, and the JDK link archives do not
# participate in whole-program Java analysis. Their byte hashes belong to the
# relink/package identity maintained by build-android-aot-sdk.sh.
#
# GraalVM does not expose a safe persistent per-JAR analysis cache. The reusable
# unit is therefore the complete, checksum-verified libsdx_llm.o produced from
# this managed-code identity.

SDX_NATIVE_IMAGE_OBJECT_STAGE_FORMAT=7
SDX_NATIVE_IMAGE_OBJECT_CACHE_SCHEMA=target-scoped-managed-content-v5
SDX_NATIVE_IMAGE_OBJECT_CONTRACT=strict-android-arm64-managed-object-v5
SDX_NATIVE_IMAGE_OBJECT_CACHE_TARGET=android-sdx-aot-object-v5
SDX_NATIVE_IMAGE_OBJECT_CACHE_ARTIFACT=libsdx_llm.o

sdx_native_image_native_archive_entry() {
  case "${1:?archive entry is required}" in
    *.so|*.so.*|*.dylib|*.dll|*.a|*.o|*.obj) return 0 ;;
    *) return 1 ;;
  esac
}

sdx_native_image_jar_analysis_manifest() {
  local archive="${1:?dependency archive is required}"
  local entry entry_sha256
  [[ -f "$archive" && ! -L "$archive" && -s "$archive" ]] || return 1
  command -v zipinfo >/dev/null 2>&1 || {
    printf 'zipinfo is required to fingerprint Native Image dependency archives\n' >&2
    return 1
  }
  command -v unzip >/dev/null 2>&1 || {
    printf 'unzip is required to fingerprint Native Image dependency archives\n' >&2
    return 1
  }

  while IFS= read -r entry; do
    [[ -n "$entry" && "$entry" != */ ]] || continue
    case "$entry" in
      META-INF/*.SF|META-INF/*.RSA|META-INF/*.DSA|META-INF/SIG-*) continue ;;
    esac
    if sdx_native_image_native_archive_entry "$entry"; then
      # Native payload bytes are linked/packaged later and cannot alter Graal's
      # managed reachability. Keep the entry name so closure changes remain
      # visible without invalidating on a rebuild of the same native library.
      printf 'native-entry\0%s\0' "$entry"
    else
      entry_sha256="$(unzip -p "$archive" "$entry" | sha256sum | cut -d ' ' -f 1)" || return 1
      printf 'content\0%s\0%s\0' "$entry" "$entry_sha256"
    fi
  done < <(LC_ALL=C zipinfo -1 "$archive" | LC_ALL=C sort)
}

sdx_native_image_jar_analysis_sha256() {
  sdx_native_image_jar_analysis_manifest "${1:?dependency archive is required}" |
    sha256sum | cut -d ' ' -f 1
}

sdx_native_image_runtime_analysis_manifest() {
  local classpath_file="${1:?effective classpath file is required}"
  local classpath_entry classpath_label entry_sha256
  local dependency_index=0

  while IFS= read -r classpath_entry; do
    [[ -n "$classpath_entry" ]] || continue
    [[ -e "$classpath_entry" ]] || {
      printf 'Native Image classpath entry is missing: %s\n' "$classpath_entry" >&2
      return 1
    }
    [[ -f "$classpath_entry" ]] || continue
    dependency_index=$((dependency_index + 1))
    classpath_label="$(basename -- "$classpath_entry")"
    case "$classpath_entry" in
      *.jar|*.zip)
        entry_sha256="$(sdx_native_image_jar_analysis_sha256 "$classpath_entry")" || return 1
        printf '%06d managed-archive %s %s\n'           "$dependency_index" "$entry_sha256" "$classpath_label"
        ;;
      *)
        entry_sha256="$(sha256sum -- "$classpath_entry" | cut -d ' ' -f 1)" || return 1
        printf '%06d managed-file %s %s\n'           "$dependency_index" "$entry_sha256" "$classpath_label"
        ;;
    esac
  done < <(tr ':' '\n' <"$classpath_file")
  [[ "$dependency_index" -gt 0 ]]
}

sdx_native_image_object_identity_lines() {
  # libsdx_llm.o contains executable code, not merely a reachability graph. Exact
  # fresh class bytes must therefore participate even when two builds have the
  # same reachable types and JNI metadata.
  printf '%s\n' \
    "format=$SDX_NATIVE_IMAGE_OBJECT_STAGE_FORMAT" \
    "cache_schema=$SDX_NATIVE_IMAGE_OBJECT_CACHE_SCHEMA" \
    "stage=android-aot-native-image-object" \
    "target=android-arm64-bionic" \
    "module=nd4j/sdx-aot" \
    "image_name=libsdx_llm.o" \
    "native_profile=cpu-managed-openblas" \
    "object_contract=$SDX_NATIVE_IMAGE_OBJECT_CONTRACT" \
    "source_manifest_sha256=$SOURCE_MANIFEST_SHA256" \
    "classes_sha256=$CLASSES_SHA256" \
    "model_classes_sha256=$MODEL_CLASSES_SHA256" \
    "fresh_class_builds_sha256=$FRESH_CLASS_BUILDS_SHA256" \
    "classpath_manifest_sha256=$CLASSPATH_MANIFEST_SHA256" \
    "runtime_analysis_manifest_sha256=$RUNTIME_ANALYSIS_MANIFEST_SHA256" \
    "maven_dependency_arguments_sha256=$MAVEN_DEPENDENCY_ARGUMENTS_SHA256" \
    "javacpp_reachability_generator_sha256=$JAVACPP_REACHABILITY_GENERATOR_SHA256" \
    "javacpp_reachability_config_sha256=$JAVACPP_REACHABILITY_CONFIG_SHA256" \
    "javacpp_jni_config_sha256=$JAVACPP_JNI_CONFIG_SHA256" \
    "javacpp_initialization_config_sha256=$JAVACPP_INITIALIZATION_CONFIG_SHA256" \
    "javacpp_reachability_manifest_sha256=$JAVACPP_REACHABILITY_MANIFEST_SHA256" \
    "native_image_build_mode=$NATIVE_IMAGE_BUILD_MODE" \
    "native_image_optimization=$NATIVE_IMAGE_OPTIMIZATION" \
    "native_image_optimization_config_sha256=$NATIVE_IMAGE_OPTIMIZATION_CONFIG_SHA256" \
    "native_image_object_identity_helper_sha256=$NATIVE_IMAGE_OBJECT_IDENTITY_HELPER_SHA256" \
    "object_builder_sha256=$OBJECT_BUILDER_SHA256" \
    "libjvm_sha256=$LIBJVM_SHA256" \
    "liblibchelper_sha256=$LIBLIBCHELPER_SHA256" \
    "maven_sha256=$MAVEN_SHA256" \
    "maven_version_sha256=$MAVEN_VERSION_SHA256" \
    "java_version_sha256=$JAVA_VERSION_SHA256" \
    "javacpp_jar_sha256=$JAVACPP_JAR_SHA256" \
    "ndk_revision_sha256=$NDK_REVISION_SHA256" \
    "graalvm_version_sha256=$GRAALVM_VERSION_SHA256" \
    "android_api=$ANDROID_API" \
    "android_abi=arm64-v8a"
}

sdx_native_image_object_identity_sha256() {
  sdx_native_image_object_identity_lines | sha256sum | cut -d ' ' -f 1
}
