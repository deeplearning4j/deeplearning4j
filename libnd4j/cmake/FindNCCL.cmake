# FindNCCL.cmake
# Finds the NVIDIA Collective Communications Library (NCCL)
#
# Sets:
#   NCCL_FOUND          - True if NCCL was found
#   NCCL_INCLUDE_DIRS   - NCCL include directories
#   NCCL_LIBRARIES      - NCCL library files
#   NCCL_VERSION        - NCCL version string (e.g., "2.18.3")
#
# Searches:
#   NCCL_ROOT           - User-specified root directory
#   CUDA_TOOLKIT_ROOT   - CUDA toolkit directory (NCCL often bundled)
#   Standard system paths
#
# Creates imported target: NCCL::nccl

include(FindPackageHandleStandardArgs)

# Build search paths
set(_NCCL_SEARCH_PATHS "")

if(DEFINED NCCL_ROOT)
    list(APPEND _NCCL_SEARCH_PATHS "${NCCL_ROOT}")
endif()

if(DEFINED ENV{NCCL_ROOT})
    list(APPEND _NCCL_SEARCH_PATHS "$ENV{NCCL_ROOT}")
endif()

if(DEFINED ENV{NCCL_HOME})
    list(APPEND _NCCL_SEARCH_PATHS "$ENV{NCCL_HOME}")
endif()

# NCCL is often bundled with CUDA toolkit
if(DEFINED CUDAToolkit_ROOT)
    list(APPEND _NCCL_SEARCH_PATHS "${CUDAToolkit_ROOT}")
endif()

if(DEFINED ENV{CUDA_HOME})
    list(APPEND _NCCL_SEARCH_PATHS "$ENV{CUDA_HOME}")
endif()

if(DEFINED ENV{CUDA_PATH})
    list(APPEND _NCCL_SEARCH_PATHS "$ENV{CUDA_PATH}")
endif()

# Standard install locations
list(APPEND _NCCL_SEARCH_PATHS
    "/usr/local/cuda"
    "/usr/local"
    "/usr"
)

# Find header
find_path(NCCL_INCLUDE_DIR
    NAMES nccl.h
    PATHS ${_NCCL_SEARCH_PATHS}
    PATH_SUFFIXES include
    NO_DEFAULT_PATH
)

# Fallback to system paths
if(NOT NCCL_INCLUDE_DIR)
    find_path(NCCL_INCLUDE_DIR NAMES nccl.h)
endif()

# Find library
find_library(NCCL_LIBRARY
    NAMES nccl
    PATHS ${_NCCL_SEARCH_PATHS}
    PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu
    NO_DEFAULT_PATH
)

# Fallback to system paths
if(NOT NCCL_LIBRARY)
    find_library(NCCL_LIBRARY NAMES nccl)
endif()

# Extract version from nccl.h
if(NCCL_INCLUDE_DIR AND EXISTS "${NCCL_INCLUDE_DIR}/nccl.h")
    file(READ "${NCCL_INCLUDE_DIR}/nccl.h" _NCCL_HEADER_CONTENTS)

    string(REGEX MATCH "#define NCCL_MAJOR[ \t]+([0-9]+)" _MATCH "${_NCCL_HEADER_CONTENTS}")
    set(_NCCL_MAJOR "${CMAKE_MATCH_1}")

    string(REGEX MATCH "#define NCCL_MINOR[ \t]+([0-9]+)" _MATCH "${_NCCL_HEADER_CONTENTS}")
    set(_NCCL_MINOR "${CMAKE_MATCH_1}")

    string(REGEX MATCH "#define NCCL_PATCH[ \t]+([0-9]+)" _MATCH "${_NCCL_HEADER_CONTENTS}")
    set(_NCCL_PATCH "${CMAKE_MATCH_1}")

    if(_NCCL_MAJOR AND _NCCL_MINOR AND _NCCL_PATCH)
        set(NCCL_VERSION "${_NCCL_MAJOR}.${_NCCL_MINOR}.${_NCCL_PATCH}")
    elseif(_NCCL_MAJOR AND _NCCL_MINOR)
        set(NCCL_VERSION "${_NCCL_MAJOR}.${_NCCL_MINOR}")
    endif()
endif()

# Standard CMake package handling
find_package_handle_standard_args(NCCL
    REQUIRED_VARS NCCL_LIBRARY NCCL_INCLUDE_DIR
    VERSION_VAR NCCL_VERSION
)

if(NCCL_FOUND)
    set(NCCL_INCLUDE_DIRS ${NCCL_INCLUDE_DIR})
    set(NCCL_LIBRARIES ${NCCL_LIBRARY})

    # Create imported target
    if(NOT TARGET NCCL::nccl)
        add_library(NCCL::nccl SHARED IMPORTED)
        set_target_properties(NCCL::nccl PROPERTIES
            IMPORTED_LOCATION "${NCCL_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${NCCL_INCLUDE_DIR}"
        )
    endif()
endif()

mark_as_advanced(NCCL_INCLUDE_DIR NCCL_LIBRARY)
