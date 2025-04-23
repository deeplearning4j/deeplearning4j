# FindARMCOMPUTE.cmake
# Find ARM Compute Library includes and libraries
#
# Sets the following variables:
#   ARMCOMPUTE_FOUND        - True if ARM Compute Library was found
#   ARMCOMPUTE_INCLUDE      - Path to ARM Compute Library include directory
#   ARMCOMPUTE_LIBRARIES    - ARM Compute Library libraries to link against
#
# This module will look for ARM Compute Library in standard locations and
# in the directory specified by the ARMCOMPUTE_ROOT variable.

# Find include directory
find_path(ARMCOMPUTE_INCLUDE
    NAMES arm_compute/core/CL/CLKernelLibrary.h
    PATHS
        ${ARMCOMPUTE_ROOT}
        ${ARMCOMPUTE_ROOT}/include
        ENV ARMCOMPUTE_ROOT
        ENV ARMCOMPUTE_PATH
        /usr/include
        /usr/local/include
)

# Find library files
find_library(ARMCOMPUTE_CORE_LIBRARY
    NAMES arm_compute_core-static arm_compute_core
    PATHS
        ${ARMCOMPUTE_ROOT}
        ${ARMCOMPUTE_ROOT}/lib
        ${ARMCOMPUTE_ROOT}/build
        ENV ARMCOMPUTE_ROOT
        ENV ARMCOMPUTE_PATH
        /usr/lib
        /usr/local/lib
)

find_library(ARMCOMPUTE_LIBRARY
    NAMES arm_compute-static arm_compute
    PATHS
        ${ARMCOMPUTE_ROOT}
        ${ARMCOMPUTE_ROOT}/lib
        ${ARMCOMPUTE_ROOT}/build
        ENV ARMCOMPUTE_ROOT
        ENV ARMCOMPUTE_PATH
        /usr/lib
        /usr/local/lib
)

# Set libraries variable
set(ARMCOMPUTE_LIBRARIES ${ARMCOMPUTE_LIBRARY} ${ARMCOMPUTE_CORE_LIBRARY})

# Handle the QUIETLY and REQUIRED arguments
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ARMCOMPUTE
    DEFAULT_MSG
    ARMCOMPUTE_INCLUDE
    ARMCOMPUTE_LIBRARIES
)

# Mark as advanced
mark_as_advanced(
    ARMCOMPUTE_INCLUDE
    ARMCOMPUTE_LIBRARY
    ARMCOMPUTE_CORE_LIBRARY
    ARMCOMPUTE_LIBRARIES
)
