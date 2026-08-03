cmake_minimum_required(VERSION 3.18)

if(NOT DEFINED SDX_SOURCE_DIR)
    message(FATAL_ERROR "SDX_SOURCE_DIR is required")
endif()

# The base classifier sees no optional backends. These values must remain AUTO
# in the cache so a later classifier can resolve them again from its HAVE_* set.
set(HAVE_TRITON OFF)
set(HAVE_ONEDNN OFF)
set(HAVE_MLIR OFF)
set(HAVE_OPENVINO OFF)
include("${SDX_SOURCE_DIR}/cmake/BuildSDX.cmake")

foreach(_feature TRITON ONEDNN MLIR OPENVINO)
    if(NOT SDX_INCLUDE_${_feature} STREQUAL "AUTO")
        message(FATAL_ERROR
            "SDX_INCLUDE_${_feature} should default to AUTO, got '${SDX_INCLUDE_${_feature}}'")
    endif()
    if(SDX_ENABLE_${_feature})
        message(FATAL_ERROR "${_feature} should be disabled when detection is OFF")
    endif()
endforeach()

# Simulate reconfiguring the same cached tree for the OneDNN classifier.
set(HAVE_ONEDNN ON)
sdx_resolve_feature_option(
    SDX_INCLUDE_ONEDNN HAVE_ONEDNN "test OneDNN option" SDX_ENABLE_ONEDNN)
if(NOT SDX_ENABLE_ONEDNN)
    message(FATAL_ERROR "AUTO did not follow HAVE_ONEDNN=ON on reconfigure")
endif()

# The CUDA compile classifier follows the same base configure and must acquire
# the complete Triton link contract instead of retaining the base OFF value.
set(HAVE_TRITON ON)
sdx_resolve_feature_option(
    SDX_INCLUDE_TRITON HAVE_TRITON "test Triton option" SDX_ENABLE_TRITON)
if(NOT SDX_ENABLE_TRITON)
    message(FATAL_ERROR "AUTO did not follow HAVE_TRITON=ON on reconfigure")
endif()

# Explicit user choices still win over dependency detection.
set(SDX_TEST_FEATURE "OFF" CACHE STRING "test override" FORCE)
set(HAVE_TEST_FEATURE ON)
sdx_resolve_feature_option(
    SDX_TEST_FEATURE HAVE_TEST_FEATURE "test feature" SDX_TEST_FEATURE_ENABLED)
if(SDX_TEST_FEATURE_ENABLED)
    message(FATAL_ERROR "explicit OFF was not honored")
endif()

set(SDX_TEST_FEATURE "ON" CACHE STRING "test override" FORCE)
set(HAVE_TEST_FEATURE OFF)
sdx_resolve_feature_option(
    SDX_TEST_FEATURE HAVE_TEST_FEATURE "test feature" SDX_TEST_FEATURE_ENABLED)
if(NOT SDX_TEST_FEATURE_ENABLED)
    message(FATAL_ERROR "explicit ON was not honored")
endif()

# The standalone SDK and every platform binding must receive the exact shared
# runtime closure published by BuildSDX (currently LLVM + MLIR for Triton).
file(READ "${SDX_SOURCE_DIR}/cmake/MainBuildFlow.cmake" _sdx_main_build_flow)
set(_sdx_packaging_contract_fragments
    [=[get_target_property(_sdx_runtime_dependency_targets]=]
    [=[SDX_RUNTIME_DEPENDENCY_TARGETS]=]
    [=[list(JOIN _sdx_sdk_shared_runtime_files "|"]=]
    [=[RUNTIME_LIBRARIES_PIPE=${_sdx_sdk_shared_runtime_files_pipe}]=]
    [=[${_sdx_shared_runtime_stage_cmds}]=]
    [=[set(_sdx_runtime_dependency_files
        ${_sdx_sdk_shared_runtime_files})]=]
    [=[SDX_RUNTIME_DEPENDENCY_FILES=${_sdx_runtime_dependency_files_encoded}]=])
foreach(_sdx_packaging_contract_fragment IN LISTS
        _sdx_packaging_contract_fragments)
    string(FIND "${_sdx_main_build_flow}"
        "${_sdx_packaging_contract_fragment}" _sdx_fragment_position)
    if(_sdx_fragment_position EQUAL -1)
        message(FATAL_ERROR
            "SDX shared runtime packaging contract is missing: ${_sdx_packaging_contract_fragment}")
    endif()
endforeach()

# The CUDA standalone target consumes nd4jcuda's object library directly.
# A cuDNN classifier therefore has to repeat the normal CUDA target's cuDNN
# link closure rather than relying on a non-transitive sibling target.
file(READ "${SDX_SOURCE_DIR}/cmake/BuildSDX.cmake" _sdx_build_contract)
set(_sdx_cudnn_link_contract_fragments
    [=[if(HAVE_CUDNN AND TARGET CUDNN::cudnn)]=]
    [=[target_link_libraries(${main_target_name} PUBLIC CUDNN::cudnn)]=]
    [=[elseif(HAVE_CUDNN AND CUDNN_LIBRARIES)]=]
    [=[target_link_libraries(${main_target_name} PUBLIC ${CUDNN_LIBRARIES})]=])
foreach(_sdx_cudnn_link_contract_fragment IN LISTS
        _sdx_cudnn_link_contract_fragments)
    string(FIND "${_sdx_build_contract}"
        "${_sdx_cudnn_link_contract_fragment}" _sdx_cudnn_fragment_position)
    if(_sdx_cudnn_fragment_position EQUAL -1)
        message(FATAL_ERROR
            "SDX cuDNN link contract is missing: ${_sdx_cudnn_link_contract_fragment}")
    endif()
endforeach()

# CMake list separators must be removed before runtime paths are placed in a
# custom-command argv entry. Otherwise a two-library LLVM/MLIR closure splits
# the $<JOIN:...> expression itself and StageSharedRuntime receives a literal.
set(_sdx_runtime_argv_contract_fragments
    [=[list(JOIN _sdx_triton_shared_runtimes "|"]=]
    [=[RUNTIME_LIBRARIES_PIPE=${_sdx_triton_shared_runtimes_pipe}]=])
foreach(_sdx_runtime_argv_contract_fragment IN LISTS
        _sdx_runtime_argv_contract_fragments)
    string(FIND "${_sdx_build_contract}"
        "${_sdx_runtime_argv_contract_fragment}" _sdx_runtime_argv_position)
    if(_sdx_runtime_argv_position EQUAL -1)
        message(FATAL_ERROR
            "SDX runtime argv encoding contract is missing: ${_sdx_runtime_argv_contract_fragment}")
    endif()
endforeach()

set(_sdx_unencoded_runtime_argv [=[RUNTIME_LIBRARIES_PIPE=$<JOIN:]=])
foreach(_sdx_contract_source IN ITEMS _sdx_build_contract _sdx_main_build_flow)
    string(FIND "${${_sdx_contract_source}}"
        "${_sdx_unencoded_runtime_argv}" _sdx_unencoded_runtime_argv_position)
    if(NOT _sdx_unencoded_runtime_argv_position EQUAL -1)
        message(FATAL_ERROR
            "SDX runtime staging still embeds a raw list in a generator expression")
    endif()
endforeach()
