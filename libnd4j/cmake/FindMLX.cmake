# FindMLX.cmake — Locate the Apple MLX C++ framework
#
# Search paths: MLX_ROOT, ENV{MLX_ROOT}, CMAKE_BINARY_DIR/mlx_install,
#               /usr/local, /opt/homebrew
#
# Output variables:
#   MLX_FOUND         — TRUE if header + library located
#   MLX_INCLUDE_DIRS  — Path to mlx/ headers
#   MLX_LIBRARIES     — Full path to libmlx library

if(MLX_FOUND)
    return()
endif()

set(_mlx_search_paths
    ${MLX_ROOT}
    $ENV{MLX_ROOT}
    "${CMAKE_BINARY_DIR}/mlx_install"
    /usr/local
    /opt/homebrew
)

# Find header
find_path(MLX_INCLUDE_DIR
    NAMES mlx/mlx.h
    PATHS ${_mlx_search_paths}
    PATH_SUFFIXES include
    NO_DEFAULT_PATH
)

# Find library (shared or static)
find_library(MLX_LIBRARY
    NAMES mlx
    PATHS ${_mlx_search_paths}
    PATH_SUFFIXES lib lib64
    NO_DEFAULT_PATH
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MLX
    REQUIRED_VARS MLX_LIBRARY MLX_INCLUDE_DIR
)

if(MLX_FOUND)
    set(MLX_INCLUDE_DIRS ${MLX_INCLUDE_DIR})
    set(MLX_LIBRARIES ${MLX_LIBRARY})
    mark_as_advanced(MLX_INCLUDE_DIR MLX_LIBRARY)
endif()
