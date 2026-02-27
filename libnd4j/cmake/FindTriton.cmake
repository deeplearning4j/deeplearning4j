# FindTriton.cmake
# Find module for pre-installed Triton GPU compiler library (libtriton).
#
# This module defines:
#   TRITON_FOUND        - TRUE if Triton is found
#   TRITON_INCLUDE_DIRS - Triton include directories
#   TRITON_LIBRARIES    - Triton libraries to link
#   TRITON_VERSION      - Triton version string
#
# Searches in:
#   TRITON_ROOT (environment or CMake variable)
#   /usr/local
#   /usr
#   ${CMAKE_BINARY_DIR}/triton_install (ExternalProject install dir)
#
# Cross-platform: works on Linux (.a) and Windows (.lib)

include(FindPackageHandleStandardArgs)

# Search paths
set(_TRITON_SEARCH_PATHS
    ${TRITON_ROOT}
    $ENV{TRITON_ROOT}
    ${CMAKE_BINARY_DIR}/triton_install
)

# Add platform-specific search paths
if(WIN32)
    # Common Windows install locations
    list(APPEND _TRITON_SEARCH_PATHS
        "$ENV{ProgramFiles}/triton"
        "$ENV{ProgramFiles\(x86\)}/triton"
        "$ENV{LOCALAPPDATA}/triton"
    )
else()
    list(APPEND _TRITON_SEARCH_PATHS
        /usr/local
        /usr
    )
endif()

# Find include directory
find_path(TRITON_INCLUDE_DIR
    NAMES triton/Compiler/Compiler.h
    PATHS ${_TRITON_SEARCH_PATHS}
    PATH_SUFFIXES include
    DOC "Triton include directory"
)

# Find main library — searches for both Unix and Windows names
find_library(TRITON_LIBRARY
    NAMES triton libtriton
    PATHS ${_TRITON_SEARCH_PATHS}
    PATH_SUFFIXES lib lib64
    DOC "Triton main library"
)

# Find backend-specific libraries (optional)
find_library(TRITON_NVIDIA_LIBRARY
    NAMES TritonNVIDIAGPU TritonNvidia
    PATHS ${_TRITON_SEARCH_PATHS}
    PATH_SUFFIXES lib lib64
    DOC "Triton NVIDIA GPU backend library"
)

find_library(TRITON_AMD_LIBRARY
    NAMES TritonAMDGPU TritonAMD
    PATHS ${_TRITON_SEARCH_PATHS}
    PATH_SUFFIXES lib lib64
    DOC "Triton AMD GPU backend library"
)

find_library(TRITON_INTEL_LIBRARY
    NAMES TritonIntelGPU TritonIntel
    PATHS ${_TRITON_SEARCH_PATHS}
    PATH_SUFFIXES lib lib64
    DOC "Triton Intel GPU backend library"
)

# Try to find version
if(TRITON_INCLUDE_DIR)
    file(GLOB _triton_version_file "${TRITON_INCLUDE_DIR}/triton/version.h")
    if(_triton_version_file)
        file(READ "${_triton_version_file}" _triton_version_content)
        string(REGEX MATCH "TRITON_VERSION \"([0-9]+\\.[0-9]+\\.[0-9]+)\"" _ "${_triton_version_content}")
        set(TRITON_VERSION "${CMAKE_MATCH_1}")
    endif()
endif()

# Standard find_package handling
find_package_handle_standard_args(Triton
    REQUIRED_VARS TRITON_LIBRARY TRITON_INCLUDE_DIR
    VERSION_VAR TRITON_VERSION
)

if(TRITON_FOUND)
    set(TRITON_INCLUDE_DIRS ${TRITON_INCLUDE_DIR})
    set(TRITON_LIBRARIES ${TRITON_LIBRARY})

    # Add backend-specific libraries if found
    if(TRITON_NVIDIA_LIBRARY)
        list(APPEND TRITON_LIBRARIES ${TRITON_NVIDIA_LIBRARY})
    endif()
    if(TRITON_AMD_LIBRARY)
        list(APPEND TRITON_LIBRARIES ${TRITON_AMD_LIBRARY})
    endif()
    if(TRITON_INTEL_LIBRARY)
        list(APPEND TRITON_LIBRARIES ${TRITON_INTEL_LIBRARY})
    endif()
endif()

mark_as_advanced(
    TRITON_INCLUDE_DIR
    TRITON_LIBRARY
    TRITON_NVIDIA_LIBRARY
    TRITON_AMD_LIBRARY
    TRITON_INTEL_LIBRARY
)
