################################################################################
#
# SDX runtime binding packaging script.
# Invoked from MainBuildFlow.cmake via:
#   cmake -D... -P SdxRuntimePackage.cmake
#
################################################################################

cmake_minimum_required(VERSION 3.15)

macro(_sdx_require var_name)
  if(NOT DEFINED ${var_name} OR "${${var_name}}" STREQUAL "")
    message(FATAL_ERROR "SdxRuntimePackage: missing required variable: ${var_name}")
  endif()
endmacro()

macro(_sdx_copy_if_exists src_path dst_dir)
  if(DEFINED ${src_path} AND NOT "${${src_path}}" STREQUAL "" AND EXISTS "${${src_path}}")
    file(COPY "${${src_path}}" DESTINATION "${dst_dir}")
  endif()
endmacro()

_sdx_require(SDX_SDK_DIR)
_sdx_require(SDX_PLATFORM_ID)
_sdx_require(SDX_OS)
_sdx_require(SDX_ARCH)
_sdx_require(SDX_VARIANT)
_sdx_require(SDX_DEFAULT_GPU_TARGET)
_sdx_require(SDX_LIBRARY_FILE)
_sdx_require(SDX_HEADER_FILE)
_sdx_require(SDX_RUNTIME_ABI)

if(NOT EXISTS "${SDX_LIBRARY_FILE}")
  message(FATAL_ERROR "SdxRuntimePackage: runtime library does not exist: ${SDX_LIBRARY_FILE}")
endif()
if(NOT EXISTS "${SDX_HEADER_FILE}")
  message(FATAL_ERROR "SdxRuntimePackage: runtime header does not exist: ${SDX_HEADER_FILE}")
endif()

set(_binding_root "${SDX_SDK_DIR}/bindings/${SDX_PLATFORM_ID}/${SDX_VARIANT}")
set(_binding_include_dir "${_binding_root}/include/dsp/runtime")
set(_binding_lib_dir "${_binding_root}/lib")
set(_binding_share_dir "${_binding_root}/share/dsp")
set(_binding_wrappers_dir "${_binding_root}/wrappers")
set(_dist_dir "${SDX_SDK_DIR}/dist")

file(MAKE_DIRECTORY "${_binding_include_dir}")
file(MAKE_DIRECTORY "${_binding_lib_dir}")
file(MAKE_DIRECTORY "${_binding_share_dir}")
file(MAKE_DIRECTORY "${_binding_wrappers_dir}")
file(MAKE_DIRECTORY "${_dist_dir}")

_sdx_copy_if_exists(SDX_HEADER_FILE "${_binding_include_dir}")
_sdx_copy_if_exists(SDX_README_FILE "${_binding_include_dir}")
_sdx_copy_if_exists(SDX_SCHEMA_FILE "${_binding_share_dir}")
_sdx_copy_if_exists(SDX_LIBRARY_FILE "${_binding_lib_dir}")

if(DEFINED SDX_BINDINGS_TEMPLATE_DIR AND
   NOT "${SDX_BINDINGS_TEMPLATE_DIR}" STREQUAL "" AND
   EXISTS "${SDX_BINDINGS_TEMPLATE_DIR}")
  file(COPY "${SDX_BINDINGS_TEMPLATE_DIR}/" DESTINATION "${_binding_wrappers_dir}")
endif()

if(DEFINED SDX_LINKER_FILE AND NOT "${SDX_LINKER_FILE}" STREQUAL "" AND
   EXISTS "${SDX_LINKER_FILE}" AND NOT "${SDX_LINKER_FILE}" STREQUAL "${SDX_LIBRARY_FILE}")
  file(COPY "${SDX_LINKER_FILE}" DESTINATION "${_binding_lib_dir}")
endif()

set(_gpu_targets_json "[]")
if("${SDX_VARIANT}" STREQUAL "cuda")
  set(_gpu_targets_json "[\"CUDA\"]")
elseif("${SDX_VARIANT}" STREQUAL "amd")
  set(_gpu_targets_json "[\"AMD\"]")
endif()

set(_features_json "")
set(_feature_pairs
    "cuda=SDX_HAVE_CUDA"
    "zluda=SDX_HAVE_ZLUDA"
    "triton=SDX_HAVE_TRITON"
    "mlir=SDX_HAVE_MLIR"
    "mlx=SDX_HAVE_MLX"
    "nnapi=SDX_HAVE_NNAPI"
    "onednn=SDX_HAVE_ONEDNN"
    "armcompute=SDX_HAVE_ARMCOMPUTE")

foreach(_feature_pair IN LISTS _feature_pairs)
  string(REPLACE "=" ";" _feature_pair_parts "${_feature_pair}")
  list(GET _feature_pair_parts 0 _feature_name)
  list(GET _feature_pair_parts 1 _feature_var)
  set(_feature_enabled "false")
  if(DEFINED ${_feature_var})
    if("${${_feature_var}}" STREQUAL "1" OR
       "${${_feature_var}}" STREQUAL "ON" OR
       "${${_feature_var}}" STREQUAL "TRUE")
      set(_feature_enabled "true")
    endif()
  endif()

  if(NOT "${_features_json}" STREQUAL "")
    string(APPEND _features_json ",")
  endif()
  string(APPEND _features_json "\n    \"${_feature_name}\": ${_feature_enabled}")
endforeach()

if(NOT "${_features_json}" STREQUAL "")
  set(_features_json "{${_features_json}\n  }")
else()
  set(_features_json "{}")
endif()

string(TIMESTAMP _created_at "%Y-%m-%dT%H:%M:%SZ" UTC)

set(_android_abi_json "null")
if(DEFINED SDX_ANDROID_ABI AND NOT "${SDX_ANDROID_ABI}" STREQUAL "")
  set(_android_abi_json "\"${SDX_ANDROID_ABI}\"")
endif()

get_filename_component(_runtime_lib_name "${SDX_LIBRARY_FILE}" NAME)
get_filename_component(_runtime_link_name "${SDX_LINKER_FILE}" NAME)
if(NOT DEFINED SDX_LINKER_FILE OR "${SDX_LINKER_FILE}" STREQUAL "" OR
   "${SDX_LINKER_FILE}" STREQUAL "${SDX_LIBRARY_FILE}")
  set(_runtime_link_name "")
endif()

set(_binding_metadata_path "${_binding_root}/binding.json")
file(WRITE "${_binding_metadata_path}" "{\n")
file(APPEND "${_binding_metadata_path}" "  \"formatVersion\": 1,\n")
file(APPEND "${_binding_metadata_path}" "  \"createdAt\": \"${_created_at}\",\n")
file(APPEND "${_binding_metadata_path}" "  \"runtimeAbi\": ${SDX_RUNTIME_ABI},\n")
file(APPEND "${_binding_metadata_path}" "  \"platform\": {\n")
file(APPEND "${_binding_metadata_path}" "    \"id\": \"${SDX_PLATFORM_ID}\",\n")
file(APPEND "${_binding_metadata_path}" "    \"os\": \"${SDX_OS}\",\n")
file(APPEND "${_binding_metadata_path}" "    \"arch\": \"${SDX_ARCH}\",\n")
file(APPEND "${_binding_metadata_path}" "    \"androidAbi\": ${_android_abi_json}\n")
file(APPEND "${_binding_metadata_path}" "  },\n")
file(APPEND "${_binding_metadata_path}" "  \"variant\": \"${SDX_VARIANT}\",\n")
file(APPEND "${_binding_metadata_path}" "  \"defaultGpuTarget\": \"${SDX_DEFAULT_GPU_TARGET}\",\n")
file(APPEND "${_binding_metadata_path}" "  \"supportedGpuTargets\": ${_gpu_targets_json},\n")
file(APPEND "${_binding_metadata_path}" "  \"runtimeLibrary\": \"${_runtime_lib_name}\",\n")
if(NOT "${_runtime_link_name}" STREQUAL "")
  file(APPEND "${_binding_metadata_path}" "  \"runtimeLinkerFile\": \"${_runtime_link_name}\",\n")
endif()
file(APPEND "${_binding_metadata_path}" "  \"features\": ${_features_json}\n")
file(APPEND "${_binding_metadata_path}" "}\n")

set(_binding_zip "${_dist_dir}/sdx-runtime-${SDX_PLATFORM_ID}-${SDX_VARIANT}.zip")
execute_process(
    COMMAND "${CMAKE_COMMAND}" -E tar "cf" "${_binding_zip}" --format=zip "."
    WORKING_DIRECTORY "${_binding_root}"
    RESULT_VARIABLE _binding_zip_rc)
if(NOT _binding_zip_rc EQUAL 0)
  message(FATAL_ERROR "SdxRuntimePackage: failed creating binding zip: ${_binding_zip}")
endif()

if(DEFINED SDX_ENABLE_ANDROID_AAR AND
   ("${SDX_ENABLE_ANDROID_AAR}" STREQUAL "1" OR
    "${SDX_ENABLE_ANDROID_AAR}" STREQUAL "ON" OR
    "${SDX_ENABLE_ANDROID_AAR}" STREQUAL "TRUE"))
  if(NOT DEFINED SDX_ANDROID_ABI OR "${SDX_ANDROID_ABI}" STREQUAL "")
    message(FATAL_ERROR "SdxRuntimePackage: Android AAR requested but SDX_ANDROID_ABI is empty")
  endif()

  set(_aar_root "${_dist_dir}/aar/${SDX_PLATFORM_ID}-${SDX_VARIANT}")
  set(_aar_headers_dir "${_aar_root}/headers/dsp/runtime")
  set(_aar_jni_dir "${_aar_root}/jni/${SDX_ANDROID_ABI}")
  set(_aar_assets_dir "${_aar_root}/assets/dsp")
  set(_aar_classes_tmp "${_aar_root}/_classes_tmp")
  set(_aar_manifest "${_aar_root}/AndroidManifest.xml")
  set(_aar_file "${_dist_dir}/sdx-runtime-${SDX_PLATFORM_ID}-${SDX_VARIANT}.aar")

  file(REMOVE_RECURSE "${_aar_root}")
  file(MAKE_DIRECTORY "${_aar_headers_dir}" "${_aar_jni_dir}" "${_aar_assets_dir}" "${_aar_classes_tmp}/META-INF")

  _sdx_copy_if_exists(SDX_HEADER_FILE "${_aar_headers_dir}")
  _sdx_copy_if_exists(SDX_LIBRARY_FILE "${_aar_jni_dir}")
  _sdx_copy_if_exists(SDX_SCHEMA_FILE "${_aar_assets_dir}")
  file(COPY "${_binding_metadata_path}" DESTINATION "${_aar_root}")

  file(WRITE "${_aar_manifest}" "<manifest xmlns:android=\"http://schemas.android.com/apk/res/android\" package=\"org.nd4j.dsp.runtime\" android:versionCode=\"1\" android:versionName=\"1.0\" />\n")
  file(WRITE "${_aar_classes_tmp}/META-INF/MANIFEST.MF" "Manifest-Version: 1.0\nCreated-By: sdx-runtime\n")

  execute_process(
      COMMAND "${CMAKE_COMMAND}" -E tar "cf" "${_aar_root}/classes.jar" --format=zip "META-INF"
      WORKING_DIRECTORY "${_aar_classes_tmp}"
      RESULT_VARIABLE _classes_jar_rc)
  if(NOT _classes_jar_rc EQUAL 0)
    message(FATAL_ERROR "SdxRuntimePackage: failed creating classes.jar for ${_aar_file}")
  endif()
  file(REMOVE_RECURSE "${_aar_classes_tmp}")

  execute_process(
      COMMAND "${CMAKE_COMMAND}" -E tar "cf" "${_aar_file}" --format=zip "."
      WORKING_DIRECTORY "${_aar_root}"
      RESULT_VARIABLE _aar_rc)
  if(NOT _aar_rc EQUAL 0)
    message(FATAL_ERROR "SdxRuntimePackage: failed creating Android AAR: ${_aar_file}")
  endif()
endif()

if(DEFINED SDX_ENABLE_APPLE_XCFRAMEWORK AND
   ("${SDX_ENABLE_APPLE_XCFRAMEWORK}" STREQUAL "1" OR
    "${SDX_ENABLE_APPLE_XCFRAMEWORK}" STREQUAL "ON" OR
    "${SDX_ENABLE_APPLE_XCFRAMEWORK}" STREQUAL "TRUE"))
  find_program(_xcodebuild_cmd xcodebuild)
  if(_xcodebuild_cmd)
    set(_xcframework_out "${_dist_dir}/ND4JDSPRuntime-${SDX_PLATFORM_ID}-${SDX_VARIANT}.xcframework")
    if(EXISTS "${_xcframework_out}")
      file(REMOVE_RECURSE "${_xcframework_out}")
    endif()

    execute_process(
        COMMAND "${_xcodebuild_cmd}" -create-xcframework
                -library "${SDX_LIBRARY_FILE}"
                -headers "${_binding_root}/include"
                -output "${_xcframework_out}"
        RESULT_VARIABLE _xcframework_rc
        OUTPUT_VARIABLE _xcframework_out_log
        ERROR_VARIABLE _xcframework_err_log)
    if(NOT _xcframework_rc EQUAL 0)
      message(WARNING "SdxRuntimePackage: xcodebuild failed for ${_xcframework_out}")
      message(WARNING "stdout: ${_xcframework_out_log}")
      message(WARNING "stderr: ${_xcframework_err_log}")
    endif()
  else()
    message(WARNING "SdxRuntimePackage: xcodebuild not available; skipping XCFramework packaging")
  endif()
endif()

message(STATUS "SdxRuntimePackage: packaged ${SDX_PLATFORM_ID}/${SDX_VARIANT} under ${_dist_dir}")
