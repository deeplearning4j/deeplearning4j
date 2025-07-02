# cmake/Setup.cmake
# Handles initial project setup and global configurations.

# Basic CMake Configuration
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS OFF)
set(CMAKE_VERBOSE_MAKEFILE OFF)

# Standard Settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
include(CheckCXXCompilerFlag)

# MSVC runtime lib can be either "MultiThreaded" or "MultiThreadedDLL", /MT and /MD respectively
set(MSVC_RT_LIB "MultiThreadedDLL")

# Initialize job pools for parallel builds
set_property(GLOBAL PROPERTY JOB_POOLS one_jobs=1 two_jobs=2)

# Set Windows specific flags
if(WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSD_WINDOWS_BUILD=true")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DSD_WINDOWS_BUILD=true")
endif()

# NOTE: SD_CPU/SD_CUDA logic, SD_LIBRARY_NAME, and DEFAULT_ENGINE 
# are now handled in the main CMakeLists.txt before this file is included

# Set optimization level based on GCC_FUNCTRACE
if(SD_GCC_FUNCTRACE)
    message("Set optimization for functrace ${SD_GCC_FUNCTRACE}")
    set(SD_OPTIMIZATION_LEVEL "0")
else()
    message("Set optimization level for no functrace ${SD_GCC_FUNCTRACE}")
    set(SD_OPTIMIZATION_LEVEL "3")
endif()
message("Set default optimization level ${SD_OPTIMIZATION_LEVEL}")
