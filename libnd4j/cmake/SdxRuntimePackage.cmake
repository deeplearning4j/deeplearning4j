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

set(_tokenizer_available 0)
if((DEFINED SDX_TOKENIZER_HEADER_FILE AND NOT "${SDX_TOKENIZER_HEADER_FILE}" STREQUAL "") OR
   (DEFINED SDX_TOKENIZER_LIBRARY_FILE AND NOT "${SDX_TOKENIZER_LIBRARY_FILE}" STREQUAL ""))
  if(NOT DEFINED SDX_TOKENIZER_HEADER_FILE OR "${SDX_TOKENIZER_HEADER_FILE}" STREQUAL "" OR
     NOT EXISTS "${SDX_TOKENIZER_HEADER_FILE}")
    message(FATAL_ERROR "SdxRuntimePackage: tokenizer header does not exist: ${SDX_TOKENIZER_HEADER_FILE}")
  endif()
  if(NOT DEFINED SDX_TOKENIZER_LIBRARY_FILE OR "${SDX_TOKENIZER_LIBRARY_FILE}" STREQUAL "" OR
     NOT EXISTS "${SDX_TOKENIZER_LIBRARY_FILE}")
    message(FATAL_ERROR "SdxRuntimePackage: tokenizer library does not exist: ${SDX_TOKENIZER_LIBRARY_FILE}")
  endif()
  set(_tokenizer_available 1)
endif()

set(_runtime_dependency_files "")
if(DEFINED SDX_RUNTIME_DEPENDENCY_FILES AND
   NOT "${SDX_RUNTIME_DEPENDENCY_FILES}" STREQUAL "")
  string(REPLACE "|" ";" _runtime_dependency_files
      "${SDX_RUNTIME_DEPENDENCY_FILES}")
endif()
# Backward-compatible single-file input for older generated build trees.
if(DEFINED SDX_RUNTIME_DEPENDENCY_FILE AND
   NOT "${SDX_RUNTIME_DEPENDENCY_FILE}" STREQUAL "")
  list(APPEND _runtime_dependency_files "${SDX_RUNTIME_DEPENDENCY_FILE}")
endif()
list(REMOVE_DUPLICATES _runtime_dependency_files)
foreach(_runtime_dependency_file IN LISTS _runtime_dependency_files)
  if(NOT EXISTS "${_runtime_dependency_file}")
    message(FATAL_ERROR
        "SdxRuntimePackage: runtime dependency does not exist: ${_runtime_dependency_file}")
  endif()
endforeach()

set(_binding_root "${SDX_SDK_DIR}/bindings/${SDX_PLATFORM_ID}/${SDX_VARIANT}")
set(_binding_include_dir "${_binding_root}/include/dsp/runtime")
set(_binding_tokenizer_include_dir "${_binding_root}/include/tokenizer")
set(_binding_lib_dir "${_binding_root}/lib")
set(_binding_share_dir "${_binding_root}/share/dsp")
set(_binding_wrappers_dir "${_binding_root}/wrappers")
set(_dist_dir "${SDX_SDK_DIR}/dist")

# Produce deterministic packages and never retain dependencies from an older
# runtime configuration.
file(REMOVE_RECURSE "${_binding_root}")
file(MAKE_DIRECTORY "${_binding_include_dir}")
file(MAKE_DIRECTORY "${_binding_lib_dir}")
file(MAKE_DIRECTORY "${_binding_share_dir}")
file(MAKE_DIRECTORY "${_binding_wrappers_dir}")
file(MAKE_DIRECTORY "${_dist_dir}")
if(_tokenizer_available)
  file(MAKE_DIRECTORY "${_binding_tokenizer_include_dir}")
endif()

_sdx_copy_if_exists(SDX_HEADER_FILE "${_binding_include_dir}")
_sdx_copy_if_exists(SDX_README_FILE "${_binding_include_dir}")
_sdx_copy_if_exists(SDX_SCHEMA_FILE "${_binding_share_dir}")
_sdx_copy_if_exists(SDX_TEXT_GENERATION_SCHEMA_FILE "${_binding_share_dir}")
_sdx_copy_if_exists(SDX_LIBRARY_FILE "${_binding_lib_dir}")
get_filename_component(_packaged_runtime_lib_name "${SDX_LIBRARY_FILE}" NAME)
set(SDX_PACKAGED_LIBRARY_FILE
    "${_binding_lib_dir}/${_packaged_runtime_lib_name}")
if(_tokenizer_available)
  _sdx_copy_if_exists(SDX_TOKENIZER_HEADER_FILE "${_binding_tokenizer_include_dir}")
  _sdx_copy_if_exists(SDX_TOKENIZER_LIBRARY_FILE "${_binding_lib_dir}")
endif()
foreach(_runtime_dependency_file IN LISTS _runtime_dependency_files)
  _sdx_copy_if_exists(_runtime_dependency_file "${_binding_lib_dir}")
endforeach()

# Strip the PACKAGED copy of the runtime library (never the build-tree
# original) to keep exported SDK artifacts small. SDX_STRIP_TOOL is the
# toolchain strip from the parent configure; empty disables stripping.
if(DEFINED SDX_STRIP_TOOL AND NOT "${SDX_STRIP_TOOL}" STREQUAL "" AND EXISTS "${SDX_STRIP_TOOL}")
  if(EXISTS "${SDX_PACKAGED_LIBRARY_FILE}")
    execute_process(
        COMMAND "${SDX_STRIP_TOOL}" --strip-unneeded
            "${SDX_PACKAGED_LIBRARY_FILE}"
        RESULT_VARIABLE _strip_rc)
    if(NOT _strip_rc EQUAL 0)
      message(WARNING
          "SdxRuntimePackage: strip failed for ${SDX_PACKAGED_LIBRARY_FILE} (rc=${_strip_rc}); shipping unstripped")
    endif()
  endif()
endif()

if(DEFINED SDX_BINDINGS_TEMPLATE_DIR AND
   NOT "${SDX_BINDINGS_TEMPLATE_DIR}" STREQUAL "" AND
   EXISTS "${SDX_BINDINGS_TEMPLATE_DIR}")
  file(COPY "${SDX_BINDINGS_TEMPLATE_DIR}/" DESTINATION "${_binding_wrappers_dir}")
endif()

# Stage the canonical Java wrapper source (kept in the nd4j-sdx module, not
# duplicated in the bindings template tree) so wrappers/java is buildable.
if(DEFINED SDX_JAVA_SRC_DIR AND NOT "${SDX_JAVA_SRC_DIR}" STREQUAL "" AND
   EXISTS "${SDX_JAVA_SRC_DIR}")
  file(MAKE_DIRECTORY "${_binding_wrappers_dir}/java/src/main/java")
  file(COPY "${SDX_JAVA_SRC_DIR}/" DESTINATION "${_binding_wrappers_dir}/java/src/main/java")
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
elseif("${SDX_VARIANT}" STREQUAL "vulkan")
  set(_gpu_targets_json "[\"VULKAN\"]")
endif()

if(NOT DEFINED SDX_DEFAULT_ACCELERATOR OR "${SDX_DEFAULT_ACCELERATOR}" STREQUAL "")
  set(SDX_DEFAULT_ACCELERATOR "NONE")
endif()
set(_accelerator_targets_json "[]")
if("${SDX_DEFAULT_ACCELERATOR}" STREQUAL "QUALCOMM_HEXAGON_HTP")
  set(_accelerator_targets_json "[\"QUALCOMM_HEXAGON_HTP\"]")
elseif("${SDX_DEFAULT_ACCELERATOR}" STREQUAL "PJRT_TPU")
  set(_accelerator_targets_json "[\"PJRT_TPU\"]")
elseif("${SDX_DEFAULT_ACCELERATOR}" STREQUAL "GOOGLE_TENSOR_TPU")
  set(_accelerator_targets_json "[\"GOOGLE_TENSOR_TPU\"]")
elseif("${SDX_DEFAULT_ACCELERATOR}" STREQUAL "NNAPI_ACCELERATOR_ONLY")
  set(_accelerator_targets_json "[\"NNAPI_ACCELERATOR_ONLY\"]")
endif()
set(_required_accelerator_device_json "null")
if(DEFINED SDX_REQUIRED_ACCELERATOR_DEVICE AND
   NOT "${SDX_REQUIRED_ACCELERATOR_DEVICE}" STREQUAL "")
  set(_required_accelerator_device_json
      "\"${SDX_REQUIRED_ACCELERATOR_DEVICE}\"")
endif()

set(_features_json "")
set(_feature_pairs
    "cuda=SDX_HAVE_CUDA"
    "zluda=SDX_HAVE_ZLUDA"
    "triton=SDX_HAVE_TRITON"
    "vulkan=SDX_HAVE_VULKAN"
    "mlir=SDX_HAVE_MLIR"
    "mlx=SDX_HAVE_MLX"
    "nnapi=SDX_HAVE_NNAPI"
    "onednn=SDX_HAVE_ONEDNN"
    "armcompute=SDX_HAVE_ARMCOMPUTE"
    "hexagon=SDX_HAVE_HEXAGON"
    "tpu=SDX_HAVE_TPU")

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
set(_tokenizer_json "{\"available\": false}")
if(_tokenizer_available)
  get_filename_component(_tokenizer_lib_name "${SDX_TOKENIZER_LIBRARY_FILE}" NAME)
  set(_tokenizer_json "{\"available\": true, \"library\": \"${_tokenizer_lib_name}\", \"header\": \"tokenizer/tokenizers_ffi.h\"}")
endif()
set(_runtime_dependency_json_items "")
foreach(_runtime_dependency_file IN LISTS _runtime_dependency_files)
  get_filename_component(_runtime_dependency_name
      "${_runtime_dependency_file}" NAME)
  if(NOT "${_runtime_dependency_json_items}" STREQUAL "")
    string(APPEND _runtime_dependency_json_items ",")
  endif()
  string(APPEND _runtime_dependency_json_items
      "\"${_runtime_dependency_name}\"")
endforeach()
set(_runtime_dependencies_json "[${_runtime_dependency_json_items}]")

# Accelerator packages publish a separate provider contract so consumers can
# cross-check the binding metadata against the runtime they are about to load.
# Keep these values centralized here: every zip/AAR produced from this binding
# root receives the same contract.
set(_provider_id "")
set(_provider_artifact_format "")
set(_provider_default_target_soc "")
set(_provider_runtime_library "")
if("${SDX_VARIANT}" STREQUAL "vulkan")
  set(_provider_id "sdx.vulkan.v1")
  set(_provider_artifact_format "sdx-vulkan-plan")
  set(_provider_default_target_soc "Android_Vulkan_1_1")
  set(_provider_runtime_library "libnd4jvulkan.so")
elseif("${SDX_VARIANT}" STREQUAL "hexagon")
  set(_provider_id "sdx.hexagon-htp.v1")
  set(_provider_artifact_format "sdx-htp-context")
  set(_provider_default_target_soc "SM8650")
  set(_provider_runtime_library "libnd4jhexagon.so")
elseif("${SDX_VARIANT}" STREQUAL "tensor-g3")
  set(_provider_id "sdx.nnapi-tensor-g3.v1")
  set(_provider_artifact_format "sdx-nnapi-plan")
  set(_provider_default_target_soc "Tensor_G3")
  set(_provider_runtime_library "libnd4jnnapi.so")
endif()
if(NOT "${_provider_id}" STREQUAL "" AND
   NOT "${_runtime_lib_name}" STREQUAL "${_provider_runtime_library}")
  message(FATAL_ERROR
      "SdxRuntimePackage: ${SDX_VARIANT} provider expects ${_provider_runtime_library}, got ${_runtime_lib_name}")
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
if(NOT "${_provider_id}" STREQUAL "")
  file(APPEND "${_binding_metadata_path}" "  \"platformProvider\": {\n")
  file(APPEND "${_binding_metadata_path}" "    \"id\": \"${_provider_id}\",\n")
  file(APPEND "${_binding_metadata_path}" "    \"abiVersion\": 1,\n")
  file(APPEND "${_binding_metadata_path}" "    \"artifactFormat\": \"${_provider_artifact_format}\",\n")
  file(APPEND "${_binding_metadata_path}" "    \"artifactFormatVersion\": 1,\n")
  file(APPEND "${_binding_metadata_path}" "    \"defaultTargetSoc\": \"${_provider_default_target_soc}\",\n")
  file(APPEND "${_binding_metadata_path}" "    \"requiresAotArtifact\": true,\n")
  file(APPEND "${_binding_metadata_path}" "    \"allowRuntimeJit\": false,\n")
  file(APPEND "${_binding_metadata_path}" "    \"allowCpuFallback\": false")
  if("${SDX_VARIANT}" STREQUAL "tensor-g3")
    file(APPEND "${_binding_metadata_path}" ",\n")
    file(APPEND "${_binding_metadata_path}" "    \"computePolicy\": \"NNAPI_EDGETPU_THEN_ACL_NEON_THEN_FUNCTIONAL_REPLAY\",\n")
    file(APPEND "${_binding_metadata_path}" "    \"placementObservable\": true,\n")
    file(APPEND "${_binding_metadata_path}" "    \"cpuExecutionMayOccur\": true\n")
  else()
    file(APPEND "${_binding_metadata_path}" "\n")
  endif()
  file(APPEND "${_binding_metadata_path}" "  },\n")
endif()
file(APPEND "${_binding_metadata_path}" "  \"defaultGpuTarget\": \"${SDX_DEFAULT_GPU_TARGET}\",\n")
file(APPEND "${_binding_metadata_path}" "  \"supportedGpuTargets\": ${_gpu_targets_json},\n")
file(APPEND "${_binding_metadata_path}" "  \"defaultAccelerator\": \"${SDX_DEFAULT_ACCELERATOR}\",\n")
file(APPEND "${_binding_metadata_path}" "  \"requiredAcceleratorDevice\": ${_required_accelerator_device_json},\n")
file(APPEND "${_binding_metadata_path}" "  \"supportedAccelerators\": ${_accelerator_targets_json},\n")
set(_standalone_json "false")
if(DEFINED SDX_STANDALONE AND ("${SDX_STANDALONE}" STREQUAL "1" OR
   "${SDX_STANDALONE}" STREQUAL "ON" OR "${SDX_STANDALONE}" STREQUAL "TRUE"))
  set(_standalone_json "true")
endif()
file(APPEND "${_binding_metadata_path}" "  \"standalone\": ${_standalone_json},\n")
file(APPEND "${_binding_metadata_path}" "  \"runtimeLibrary\": \"${_runtime_lib_name}\",\n")
if(NOT "${_runtime_link_name}" STREQUAL "")
  file(APPEND "${_binding_metadata_path}" "  \"runtimeLinkerFile\": \"${_runtime_link_name}\",\n")
endif()
file(APPEND "${_binding_metadata_path}" "  \"runtimeDependencies\": ${_runtime_dependencies_json},\n")
file(APPEND "${_binding_metadata_path}" "  \"tokenizer\": ${_tokenizer_json},\n")
file(APPEND "${_binding_metadata_path}" "  \"features\": ${_features_json}\n")
file(APPEND "${_binding_metadata_path}" "}\n")

set(_provider_metadata_path "")
if(NOT "${_provider_id}" STREQUAL "")
  set(_provider_metadata_path "${_binding_root}/provider.json")
  file(WRITE "${_provider_metadata_path}" "{\n")
  file(APPEND "${_provider_metadata_path}" "  \"formatVersion\": 1,\n")
  file(APPEND "${_provider_metadata_path}" "  \"coreAbi\": ${SDX_RUNTIME_ABI},\n")
  file(APPEND "${_provider_metadata_path}" "  \"providerId\": \"${_provider_id}\",\n")
  file(APPEND "${_provider_metadata_path}" "  \"providerAbi\": 1,\n")
  file(APPEND "${_provider_metadata_path}" "  \"artifactFormat\": \"${_provider_artifact_format}\",\n")
  file(APPEND "${_provider_metadata_path}" "  \"artifactFormatVersion\": 1,\n")
  file(APPEND "${_provider_metadata_path}" "  \"platformId\": \"${SDX_PLATFORM_ID}\",\n")
  file(APPEND "${_provider_metadata_path}" "  \"architecture\": \"${SDX_ARCH}\",\n")
  file(APPEND "${_provider_metadata_path}" "  \"defaultTargetSoc\": \"${_provider_default_target_soc}\",\n")
  file(APPEND "${_provider_metadata_path}" "  \"runtimeLibrary\": \"${_provider_runtime_library}\",\n")
  file(APPEND "${_provider_metadata_path}" "  \"requiresAotArtifact\": true,\n")
  file(APPEND "${_provider_metadata_path}" "  \"allowRuntimeJit\": false,\n")
  file(APPEND "${_provider_metadata_path}" "  \"allowCpuFallback\": false\n")
  file(APPEND "${_provider_metadata_path}" "}\n")
endif()

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
  set(_aar_tokenizer_headers_dir "${_aar_root}/headers/tokenizer")
  set(_aar_jni_dir "${_aar_root}/jni/${SDX_ANDROID_ABI}")
  set(_aar_assets_dir "${_aar_root}/assets/dsp")
  set(_aar_classes_tmp "${_aar_root}/_classes_tmp")
  set(_aar_manifest "${_aar_root}/AndroidManifest.xml")
  set(_aar_metadata_dir "${_aar_root}/META-INF/com/android/build/gradle")
  set(_prefab_root "${_aar_root}/prefab")
  set(_prefab_module_dir "${_prefab_root}/modules/sdx_runtime")
  set(_prefab_include_dir "${_prefab_module_dir}/include/dsp/runtime")
  set(_prefab_abi_dir "${_prefab_module_dir}/libs/android.${SDX_ANDROID_ABI}")
  set(_prefab_tokenizer_module_dir "${_prefab_root}/modules/sdx_tokenizer")
  set(_prefab_tokenizer_include_dir "${_prefab_tokenizer_module_dir}/include")
  set(_prefab_tokenizer_abi_dir "${_prefab_tokenizer_module_dir}/libs/android.${SDX_ANDROID_ABI}")
  set(_aar_file "${_dist_dir}/sdx-runtime-${SDX_PLATFORM_ID}-${SDX_VARIANT}.aar")

  if(NOT DEFINED SDX_ANDROID_API OR "${SDX_ANDROID_API}" STREQUAL "")
    set(SDX_ANDROID_API 24)
  endif()
  if(NOT DEFINED SDX_ANDROID_NDK_MAJOR OR "${SDX_ANDROID_NDK_MAJOR}" STREQUAL "")
    set(SDX_ANDROID_NDK_MAJOR 0)
  endif()
  if(NOT DEFINED SDX_ANDROID_STL OR "${SDX_ANDROID_STL}" STREQUAL "")
    set(SDX_ANDROID_STL "c++_shared")
  endif()
  get_filename_component(_prefab_library_name "${SDX_LIBRARY_FILE}" NAME_WE)
  if(_tokenizer_available)
    get_filename_component(_prefab_tokenizer_library_name "${SDX_TOKENIZER_LIBRARY_FILE}" NAME_WE)
  endif()

  file(REMOVE_RECURSE "${_aar_root}")
  file(MAKE_DIRECTORY
      "${_aar_headers_dir}"
      "${_aar_jni_dir}"
      "${_aar_assets_dir}"
      "${_aar_classes_tmp}/META-INF"
      "${_aar_metadata_dir}"
      "${_prefab_include_dir}"
      "${_prefab_abi_dir}")
  if(_tokenizer_available)
    file(MAKE_DIRECTORY
        "${_aar_tokenizer_headers_dir}"
        "${_prefab_tokenizer_include_dir}"
        "${_prefab_tokenizer_abi_dir}")
  endif()

  _sdx_copy_if_exists(SDX_HEADER_FILE "${_aar_headers_dir}")
  _sdx_copy_if_exists(SDX_HEADER_FILE "${_prefab_include_dir}")
  # Use the stripped binding copy, not the multi-gigabyte build-tree binary.
  _sdx_copy_if_exists(SDX_PACKAGED_LIBRARY_FILE "${_aar_jni_dir}")
  _sdx_copy_if_exists(SDX_PACKAGED_LIBRARY_FILE "${_prefab_abi_dir}")
  if(_tokenizer_available)
    _sdx_copy_if_exists(SDX_TOKENIZER_HEADER_FILE "${_aar_tokenizer_headers_dir}")
    _sdx_copy_if_exists(SDX_TOKENIZER_HEADER_FILE "${_prefab_tokenizer_include_dir}")
    _sdx_copy_if_exists(SDX_TOKENIZER_LIBRARY_FILE "${_aar_jni_dir}")
    _sdx_copy_if_exists(SDX_TOKENIZER_LIBRARY_FILE "${_prefab_tokenizer_abi_dir}")
  endif()
  foreach(_runtime_dependency_file IN LISTS _runtime_dependency_files)
    _sdx_copy_if_exists(_runtime_dependency_file "${_aar_jni_dir}")
  endforeach()
  _sdx_copy_if_exists(SDX_SCHEMA_FILE "${_aar_assets_dir}")
  _sdx_copy_if_exists(SDX_TEXT_GENERATION_SCHEMA_FILE "${_aar_assets_dir}")
  file(COPY "${_binding_metadata_path}" DESTINATION "${_aar_root}")
  if(NOT "${_provider_metadata_path}" STREQUAL "")
    file(COPY "${_provider_metadata_path}" DESTINATION "${_aar_root}")
  endif()

  # Android Prefab turns the AAR into a normal CMake dependency:
  # find_package(sdx_runtime CONFIG REQUIRED) + sdx_runtime::sdx_runtime.
  file(WRITE "${_prefab_root}/prefab.json"
      "{\n  \"schema_version\": 2,\n  \"name\": \"sdx-runtime-${SDX_VARIANT}\",\n  \"version\": \"1.0.0\",\n  \"dependencies\": []\n}\n")
  file(WRITE "${_prefab_module_dir}/module.json"
      "{\n  \"library_name\": \"${_prefab_library_name}\",\n  \"export_libraries\": []\n}\n")
  file(WRITE "${_prefab_abi_dir}/abi.json"
      "{\n  \"abi\": \"${SDX_ANDROID_ABI}\",\n  \"api\": ${SDX_ANDROID_API},\n  \"ndk\": ${SDX_ANDROID_NDK_MAJOR},\n  \"stl\": \"${SDX_ANDROID_STL}\",\n  \"static\": false\n}\n")
  if(_tokenizer_available)
    file(WRITE "${_prefab_tokenizer_module_dir}/module.json"
        "{\n  \"library_name\": \"${_prefab_tokenizer_library_name}\",\n  \"export_libraries\": []\n}\n")
    file(WRITE "${_prefab_tokenizer_abi_dir}/abi.json"
        "{\n  \"abi\": \"${SDX_ANDROID_ABI}\",\n  \"api\": ${SDX_ANDROID_API},\n  \"ndk\": ${SDX_ANDROID_NDK_MAJOR},\n  \"stl\": \"none\",\n  \"static\": false\n}\n")
  endif()

  file(WRITE "${_aar_metadata_dir}/aar-metadata.properties"
      "aarFormatVersion=1.0\naarMetadataVersion=1.0\nminCompileSdk=${SDX_ANDROID_API}\nminCompileSdkExtension=0\n")
  file(WRITE "${_aar_manifest}"
      "<manifest xmlns:android=\"http://schemas.android.com/apk/res/android\" package=\"org.nd4j.dsp.runtime\" android:versionCode=\"1\" android:versionName=\"1.0\"><uses-sdk android:minSdkVersion=\"${SDX_ANDROID_API}\" /></manifest>\n")
  file(WRITE "${_aar_root}/consumer-rules.pro" "# Native-only SDX Prefab AAR; no Java symbols to shrink.\n")
  file(WRITE "${_aar_root}/proguard.txt" "# Native-only SDX Prefab AAR.\n")
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

    if(_tokenizer_available)
      set(_tokenizer_xcframework_out "${_dist_dir}/ND4JTokenizer-${SDX_PLATFORM_ID}.xcframework")
      if(EXISTS "${_tokenizer_xcframework_out}")
        file(REMOVE_RECURSE "${_tokenizer_xcframework_out}")
      endif()
      execute_process(
          COMMAND "${_xcodebuild_cmd}" -create-xcframework
                  -library "${SDX_TOKENIZER_LIBRARY_FILE}"
                  -headers "${_binding_tokenizer_include_dir}"
                  -output "${_tokenizer_xcframework_out}"
          RESULT_VARIABLE _tokenizer_xcframework_rc
          OUTPUT_VARIABLE _tokenizer_xcframework_out_log
          ERROR_VARIABLE _tokenizer_xcframework_err_log)
      if(NOT _tokenizer_xcframework_rc EQUAL 0)
        message(WARNING "SdxRuntimePackage: xcodebuild failed for ${_tokenizer_xcframework_out}")
        message(WARNING "stdout: ${_tokenizer_xcframework_out_log}")
        message(WARNING "stderr: ${_tokenizer_xcframework_err_log}")
      endif()
    endif()
  else()
    message(WARNING "SdxRuntimePackage: xcodebuild not available; skipping XCFramework packaging")
  endif()
endif()

message(STATUS "SdxRuntimePackage: packaged ${SDX_PLATFORM_ID}/${SDX_VARIANT} under ${_dist_dir}")
