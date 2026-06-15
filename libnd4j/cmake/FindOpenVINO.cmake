# FindOpenVINO.cmake — Locate the OpenVINO C++ runtime
#
# Uses find_package(OpenVINO) first (CMake config mode from OpenVINO install),
# then falls back to manual header/library search.
#
# Search paths: OpenVINO_ROOT, ENV{INTEL_OPENVINO_DIR}, /opt/intel/openvino,
#               /usr/local, CMAKE_BINARY_DIR/openvino_install
#
# Output variables:
#   OpenVINO_FOUND        — TRUE if header + library located
#   OPENVINO_INCLUDE_DIRS — Path to openvino/ headers
#   OPENVINO_LIBRARIES    — Libraries to link against

if(OpenVINO_FOUND)
    return()
endif()

# Build search paths list
set(_ov_search_paths
    ${OpenVINO_ROOT}
    $ENV{INTEL_OPENVINO_DIR}
    $ENV{OpenVINO_ROOT}
    "${CMAKE_BINARY_DIR}/openvino_install"
    "${CMAKE_BINARY_DIR}/openvino_install/runtime"
    /opt/intel/openvino
    /opt/intel/openvino/runtime
    /usr/local
)

# Build cmake config search paths
set(_ov_cmake_paths)
foreach(_p ${_ov_search_paths})
    list(APPEND _ov_cmake_paths
        "${_p}/lib/cmake/OpenVINO"
        "${_p}/runtime/lib/cmake/OpenVINO"
        "${_p}/runtime/cmake"
        "${_p}/cmake"
    )
endforeach()

# Try CMake config mode first (preferred — installed OpenVINO provides OpenVINOConfig.cmake)
find_package(OpenVINO QUIET CONFIG PATHS ${_ov_cmake_paths} NO_DEFAULT_PATH)
if(NOT OpenVINO_FOUND)
    find_package(OpenVINO QUIET CONFIG)
endif()

if(OpenVINO_FOUND)
    # OpenVINO config mode sets up imported targets: openvino::runtime
    set(OPENVINO_INCLUDE_DIRS "")  # handled by imported target
    set(OPENVINO_LIBRARIES openvino::runtime)
    message(STATUS "Found OpenVINO via CMake config mode")
    return()
endif()

# Fallback: manual search
find_path(OPENVINO_INCLUDE_DIR
    NAMES openvino/openvino.hpp
    PATHS ${_ov_search_paths}
    PATH_SUFFIXES include runtime/include
    NO_DEFAULT_PATH
)

# Find library — try both shared and static
find_library(OPENVINO_LIBRARY
    NAMES openvino
    PATHS ${_ov_search_paths}
    PATH_SUFFIXES lib lib64 lib/intel64 lib/x86_64-linux-gnu
                  runtime/lib runtime/lib/intel64 runtime/lib64
    NO_DEFAULT_PATH
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OpenVINO
    REQUIRED_VARS OPENVINO_LIBRARY OPENVINO_INCLUDE_DIR
)

if(OpenVINO_FOUND)
    set(OPENVINO_INCLUDE_DIRS ${OPENVINO_INCLUDE_DIR})
    set(OPENVINO_LIBRARIES ${OPENVINO_LIBRARY})
    mark_as_advanced(OPENVINO_INCLUDE_DIR OPENVINO_LIBRARY)
endif()
