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
