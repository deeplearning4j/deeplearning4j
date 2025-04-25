################################################################################
#
# Apache License, Version 2.0 - See LICENSE file and NOTICE file
# SPDX-License-Identifier: Apache-2.0
#
################################################################################

cmake_minimum_required(VERSION 3.15)
project(libnd4j)

set(CMAKE_VERBOSE_MAKEFILE ON) # Changed from OFF in blas/CMakeLists.txt to ON from main CMakeLists.txt
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Module Path (from main)
set (CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
message("CMAKE MODULE PATH ${CMAKE_MODULE_PATH}")

# Windows Export Symbols (from main)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS OFF)
if (WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSD_WINDOWS_BUILD=true")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DSD_WINDOWS_BUILD=true")
endif()

# C++/CUDA Standard (from main)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Options (merged from main and blas - blas options seem mostly superseded or handled differently in main)
option(SD_NATIVE "Optimize for build machine (might not work on others)" OFF)
option(SD_CHECK_VECTORIZATION "checks for vectorization" OFF)
option(SD_BUILD_TESTS "Build tests" OFF)
option(SD_STATIC_LIB "Build static library" OFF)
option(SD_SHARED_LIB "Build shared library" ON)
option(SD_SANITIZE "Enable Address Sanitizer" OFF) # Added SD_SANITIZERS variable for flexibility
option(SD_USE_LTO "Use link time optimization" OFF)
option(PRINT_INDICES "Print indices" OFF)
option(PRINT_MATH "Print math operations" OFF)
option(SD_PREPROCESS "Use preprocessor" OFF) # Renamed from SD_PREPROCESSOR in main
option(SD_GCC_FUNCTRACE "Use call traces" OFF)
option(FLATBUFFERS_BUILD_FLATC "Enable the build of the flatbuffers compiler" OFF) # Default from main, but can be overridden by env var later
option(SD_PTXAS "Enable PTXAS verbose output" OFF) # Added option for clarity
option(SD_KEEP_NVCC_OUTPUT "Keep intermediate NVCC files" OFF) # Added option for clarity

# Include CheckCXXCompilerFlag (from blas)
include(CheckCXXCompilerFlag)

# === Settings from blas/CMakeLists.txt ===

# Include Directories Based on OS
if(UNIX AND NOT APPLE) # Refined condition
    link_directories(/usr/local/lib /usr/lib /lib)
endif()
if(APPLE)
    message("Using Apple")
    link_directories(/usr/local/lib /usr/lib /lib)
endif()

# Define Compiler Flags for Specific Builds (Apple, ARM, Android, iOS)
if(SD_APPLE_BUILD)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSD_APPLE_BUILD=true -mmacosx-version-min=10.10")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DSD_APPLE_BUILD=true -mmacosx-version-min=10.10")
endif()
if(SD_ARM_BUILD)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSD_ARM_BUILD=true")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DSD_ARM_BUILD=true")
endif()
if(SD_ANDROID_BUILD)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSD_ANDROID_BUILD=true")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DSD_ANDROID_BUILD=true")
endif()
if(SD_IOS_BUILD)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSD_IOS_BUILD=true")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DSD_IOS_BUILD=true")
endif()

# Windows Specific Configurations (from blas)
if(WIN32 AND NOT ANDROID)
    get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES) # Using CMAKE_CURRENT_SOURCE_DIR assuming this is now top-level
    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wa,-mbig-obj")
    endif()

    message(STATUS "Include Directories:")
    foreach(dir ${dirs})
        message(STATUS "dir='${dir}'")
    endforeach()

    # Workaround for Long Command Lines
    set(CMAKE_C_USE_RESPONSE_FILE_FOR_OBJECTS ON)
    set(CMAKE_CXX_USE_RESPONSE_FILE_FOR_OBJECTS ON)
    set(CMAKE_C_RESPONSE_FILE_LINK_FLAG "@")
    set(CMAKE_CXX_RESPONSE_FILE_LINK_FLAG "@")
    set(CMAKE_NINJA_FORCE_RESPONSE_FILE ON CACHE INTERNAL "")
endif()

# Link Time Optimization (from blas)
if(SD_USE_LTO)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        message(STATUS "Using Link Time Optimization")
        add_compile_options(-flto)
        add_link_options(-flto)
    endif()
endif()

# === End settings from blas/CMakeLists.txt ===

# Print Indices/Math Flags (from main) [cite: 1]
message("PRINT_INDICES: ${PRINT_INDICES}")
if("${PRINT_INDICES}" STREQUAL "ON")
    message("Added print indices compile definition")
    add_compile_definitions(PRINT_INDICES)
endif()

message("PRINT_MATH: ${PRINT_MATH}")
if("${PRINT_MATH}" STREQUAL "ON")
    message("Added print math compile definition") # Corrected message [cite: 1]
    add_compile_definitions(SD_PRINT_MATH)
endif()

# Optimization Level (from main)
if("${SD_GCC_FUNCTRACE}" STREQUAL "ON")
    message("Set optimization for functrace ${SD_GCC_FUNCTRACE}")
    set(SD_OPTIMIZATION_LEVEL   "0")

else()
    message("Set optimization level for no functrace ${SD_GCC_FUNCTRACE}")
    set(SD_OPTIMIZATION_LEVEL  "3")
endif()
message("Set default optimization level ${SD_OPTIMIZATION_LEVEL}")

# Hack to disable flatc build (from main)
set(FLATBUFFERS_BUILD_FLATC "OFF" CACHE STRING "Hack to disable flatc build" FORCE)

# Build Type Info (from main) [cite: 3, 4, 5, 6, 7, 8]
message("BUILD TYPE: ${CMAKE_BUILD_TYPE}") # Note: BUILD_TYPE logic for functrace is handled later

# Helper macro to print variables (from main) [cite: 8]
macro(print_all_variables)
    message(STATUS "print_all_variables------------------------------------------{")
    get_cmake_property(_variableNames VARIABLES)
    foreach (_variableName ${_variableNames})
        message(STATUS "${_variableName}=${${_variableName}}")
    endforeach()
    message(STATUS "print_all_variables------------------------------------------}")
endmacro()
# print_all_variables() # Uncomment to debug

# --- Platform/Arch Detection (Consolidated from main and blas) ---
set(SD_X86_BUILD false)
set(SD_ARM_BUILD false)

# Default Arch if not provided externally
if(NOT DEFINED SD_ARCH)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
        set(SD_ARCH "x86-64")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm*|aarch64")
        # Need more specific detection, default to arm64-v8a for now if arm*
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
            set(SD_ARCH "arm64-v8a") # More specific for aarch64
        else()
            set(SD_ARCH "armv7-a") # Default assumption for generic 'arm'
        endif()
    else()
        set(SD_ARCH ${CMAKE_SYSTEM_PROCESSOR}) # Fallback
    endif()
    message(STATUS "SD_ARCH not defined externally, detected as: ${SD_ARCH}")
endif()

# Determine platform type based on SD_ARCH and build flags
if(SD_ANDROID_BUILD)
    if(ANDROID_ABI MATCHES "x86_64")
        set(SD_X86_BUILD true)
        set(SD_ARCH "x86-64") # Override SD_ARCH based on ABI
    elseif(ANDROID_ABI MATCHES "x86")
        set(SD_X86_BUILD true)
        set(SD_ARCH "x86") # Override SD_ARCH based on ABI
    elseif(ANDROID_ABI MATCHES "arm64-v8a")
        set(SD_ARM_BUILD true)
        set(SD_ARCH "arm64-v8a") # Override SD_ARCH based on ABI [cite: 77]
    elseif(ANDROID_ABI MATCHES "armeabi-v7a")
        set(SD_ARM_BUILD true)
        set(SD_ARCH "armv7-a") # Override SD_ARCH based on ABI [cite: 78]
    endif()
elseif(APPLE)
    # Apple specific checks
    if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm64*" OR "${SD_ARCH}" MATCHES "armv8-a" OR "${SD_ARCH}" MATCHES "arm64")
        set(SD_ARM_BUILD true)
        set(SD_X86_BUILD false)
        set(SD_ARCH "arm64") # Standardize Apple ARM arch
        set(CMAKE_OSX_ARCHITECTURES "arm64")
    else() # Assume x86_64 for non-ARM Apple
        set(SD_X86_BUILD true)
        set(SD_ARM_BUILD false)
        set(SD_ARCH "x86-64")
        set(CMAKE_OSX_ARCHITECTURES "x86_64")
    endif()
elseif(NOT SD_IOS_BUILD) # Generic Non-Android, Non-Apple, Non-iOS
    if(${SD_ARCH} MATCHES "x86_64|x86")
        set(SD_X86_BUILD true)
        set(SD_ARM_BUILD false)
    elseif(${SD_ARCH} MATCHES "arm*|aarch*") [cite: 79]
set(SD_X86_BUILD false)
set(SD_ARM_BUILD true)
# Add other architectures like PowerPC if needed
elseif(${SD_ARCH} MATCHES "power*")
set(SD_X86_BUILD false)
set(SD_ARM_BUILD false)
# No specific powerpc flag needed currently
endif()
endif()

message(STATUS "Build flags determined: SD_ANDROID_BUILD=${SD_ANDROID_BUILD}, SD_IOS_BUILD=${SD_IOS_BUILD}, SD_APPLE_BUILD=${SD_APPLE_BUILD}, SD_X86_BUILD=${SD_X86_BUILD}, SD_ARM_BUILD=${SD_ARM_BUILD}, SD_ARCH=${SD_ARCH}")

# --- ARM Compute Setup (from main) ---
# Define ARM Compute Library URLs based on architecture
set(ARM_COMPUTE_URL_ARMV7 "https://github.com/ARM-software/ComputeLibrary/releases/download/v25.04/arm_compute-v25.04-linux-armv7a-cpu-bin.tar.gz") [cite: 8]
set(ARM_COMPUTE_URL_AARCH64 "https://github.com/ARM-software/ComputeLibrary/releases/download/v25.04/arm_compute-v25.04-linux-aarch64-cpu-bin.tar.gz") [cite: 8]
set(ARM_COMPUTE_URL_ANDROID_AARCH64 "https://github.com/ARM-software/ComputeLibrary/releases/download/v25.04/arm_compute-v25.04-android-aarch64-cpu-bin.tar.gz") [cite: 8]

# Set the appropriate URL based on the detected architecture
if(${SD_ARCH} MATCHES "armv7")
set(ARM_COMPUTE_URL ${ARM_COMPUTE_URL_ARMV7})
message("Using ARM Compute Library for ARMv7")
elseif((${SD_ARCH} MATCHES "armv8-a") OR (${SD_ARCH} MATCHES "arm64")) # Combined condition [cite: 9]
if(SD_ANDROID_BUILD)
set(ARM_COMPUTE_URL ${ARM_COMPUTE_URL_ANDROID_AARCH64})
message("Using ARM Compute Library for Android ARM64 (matched ${SD_ARCH})") [cite: 9, 10]
else()
set(ARM_COMPUTE_URL ${ARM_COMPUTE_URL_AARCH64})
message("Using ARM Compute Library for Linux ARM64 (matched ${SD_ARCH})") [cite: 9, 10]
endif()
endif()

# --- Build Type Flags (Consolidated from main and blas) ---
set(MSVC_RT_LIB "MultiThreadedDLL") # Default, might be adjusted later

if (SD_ANDROID_BUILD)
# Android flags (from main)
set_property(GLOBAL PROPERTY JOB_POOLS one_job=1 two_jobs=2)
set(CMAKE_CXX_FLAGS_RELEASE  "${CMAKE_CXX_FLAGS_RELEASE} -O${SD_OPTIMIZATION_LEVEL} -fPIC -Wno-return-type -Wno-unknown-pragmas -Wno-braced-scalar-init -Wno-delete-non-virtual-dtor -Wno-unused-command-line-argument -Wno-dangling-else -D_RELEASE=true")
set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} -O${SD_OPTIMIZATION_LEVEL} -g -fPIC -Wno-return-type -Wno-unknown-pragmas -Wno-braced-scalar-init -Wno-delete-non-virtual-dtor -Wno-unused-command-line-argument -Wno-dangling-else") [cite: 80]
elseif (APPLE)
# Apple flags (from main)
set(CMAKE_CXX_FLAGS_RELEASE  "${CMAKE_CXX_FLAGS_RELEASE} -O${SD_OPTIMIZATION_LEVEL} -fPIC -Wno-return-type -Wno-braced-scalar-init -Wno-unknown-pragmas -Wno-delete-non-virtual-dtor -Wno-unused-command-line-argument -Wno-dangling-else -D__APPLE_OS__=true -D_RELEASE=true")
set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} -O${SD_OPTIMIZATION_LEVEL} -g -fPIC -Wno-return-type -Wno-braced-scalar-init -Wno-unknown-pragmas -Wno-delete-non-virtual-dtor -Wno-unused-command-line-argument -Wno-dangling-else -D__APPLE_OS__=true")
elseif(WIN32)
# Windows flags (from main and blas)
# Use /EHsc for both Debug and Release on MSVC
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc /bigobj") # Add /bigobj from blas [cite: 43]
# Set standard based on CUDA presence
if(SD_CUDA)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++14") # from blas CUDA section [cite: 43]
else()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++17") # Default C++17 from main
endif()
# Release/Debug specific flags
set(CMAKE_CXX_FLAGS_RELEASE  "${CMAKE_CXX_FLAGS_RELEASE} -D_RELEASE=true")
set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} /FS") # Keep /FS from main [cite: 81]
# Set MSVC Runtime Lib based on build type - Applied later to targets
else() # Linux/Other UNIX (non-Android, non-Apple)
# Base flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC") # Ensure fPIC
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT SD_CUDA)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11") # From blas, consider C++17 from main? Let's keep 11 for now as per blas for non-CUDA GNU
elseif(NOT CMAKE_CXX_COMPILER_ID STREQUAL "GNU") # Clang, Intel etc.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17") # Use C++17 from main
endif()

# Linker flags (from blas)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT CMAKE_SYSTEM_NAME MATCHES "Aurora" AND NOT SD_CUDA)
if(UNIX)
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-rpath,$ORIGIN/,-z,--no-undefined,--verbose")
else() # Should this be WIN32? Blas had this else without condition
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-rpath,$ORIGIN/,--no-undefined,--verbose")
endif()
if(CMAKE_BUILD_TYPE STREQUAL "Debug" AND NOT APPLE AND NOT WIN32)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -rdynamic -Wl,-export-dynamic,--verbose")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -export-dynamic,--verbose")
endif()
endif()

# Functrace flags (merged from main and blas) [cite: 2, 82, 83, 84, 85, 86, 87]
if("${SD_GCC_FUNCTRACE}" STREQUAL "ON")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fstack-protector -fstack-protector-all -Wall -Wextra -Werror -Wno-return-type -Wno-error=int-in-bool-context -Wno-unused-variable -Wno-error=implicit-fallthrough -Wno-unused-parameter -Wno-error=unknown-pragmas -ggdb3 -lpthread -pthread -MT -Bsymbolic -lbfd -rdynamic -lunwind -ldw -ldl -fno-omit-frame-pointer -fno-optimize-sibling-calls -finstrument-functions -O0") # Combined flags from blas, main sets CMAKE_CXX_FLAGS_RELEASE separately
add_compile_definitions(SD_GCC_FUNCTRACE)
# Apply specifically to Release/Debug as per main's logic
set(CMAKE_CXX_FLAGS_RELEASE   "${CMAKE_CXX_FLAGS_RELEASE} -O${SD_OPTIMIZATION_LEVEL} -fPIC -g") # From main [cite: 87]
set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} -g -O${SD_OPTIMIZATION_LEVEL} -fPIC") # From main
else()
set(CMAKE_CXX_FLAGS_RELEASE   "${CMAKE_CXX_FLAGS_RELEASE} -O${SD_OPTIMIZATION_LEVEL} -fPIC -D_RELEASE=true") # From main [cite: 87]
set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} -g -O${SD_OPTIMIZATION_LEVEL} -fPIC") # From main
endif()

# Sanitizer flags (from main) [cite: 88, 89]
if (SD_SANITIZE)
# Default sanitizers if not specified
if(NOT DEFINED SD_SANITIZERS)
set(SD_SANITIZERS "address,undefined,float-divide-by-zero,float-cast-overflow")
endif()
set(SANITIZE_FLAGS " -Wall -Wextra -fPIE -lpthread -ftls-model=local-dynamic -static-libasan -fsanitize=${SD_SANITIZERS} -fno-sanitize-recover=all")
message("Using sanitizers: ${SD_SANITIZERS} - Note potential conflicts (address/thread, address/undefined). See: https://github.com/google/sanitizers/issues/856") [cite: 88]
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SANITIZE_FLAGS}")
if(SD_CUDA) # Add CUDA specific sanitize flags [cite: 89]
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --relocatable-device-code=true") # Add relocatable device code for CUDA + sanitize
# Note: CUDA sanitizer flags might need specific handling depending on the sanitizer used (e.g., compute-sanitizer)
endif()
endif()

# OpenMP for GCC (from blas CPU section)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
find_package(OpenMP)
if(OPENMP_FOUND)
message(STATUS "OpenMP found, enabling flags.")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
# Link OpenMP libraries later via target_link_libraries
set(OpenMP_LIBRARIES ${OpenMP_CXX_LIBRARIES}) # Store for linking
else()
message(WARNING "OpenMP not found, parallel execution might be limited.")
endif()
endif()
endif()

# Native optimization flag (from main)
if(SD_NATIVE)
message(STATUS "Enabling native optimizations.")
IF(${CMAKE_SYSTEM_PROCESSOR} MATCHES "ppc64*" OR ${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm64*")
set(SD_X86_BUILD false) # Ensure consistency [cite: 89]
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcpu=native")
ELSE() # Assume x86 or similar
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
ENDIF()
endif()


# Architecture Tuning (from blas) - Applied after general flags
set(ARCH_TUNE "")
if(SD_ARCH MATCHES "armv8")
set(ARCH_TUNE "-march=${SD_ARCH}")
elseif(SD_ARCH MATCHES "armv7")
set(ARCH_TUNE "-march=${SD_ARCH} -mfpu=neon")
elseif(CMAKE_SYSTEM_NAME MATCHES "Aurora")
# Special case for Aurora - Apply flag later using set_source_files_properties if needed
message(STATUS "Aurora system detected, potential specific flags needed.")
elseif(SD_ARCH MATCHES "power*")
set(ARCH_TUNE "-mcpu=${SD_ARCH} -mtune=${SD_ARCH} -D__POWER")
elseif(SD_EXTENSION MATCHES "avx2") # SD_EXTENSION needs to be defined, e.g., via cmake -DSD_EXTENSION=avx2 ..
message("Building AVX2 binary...")
set(ARCH_TUNE "-mmmx -msse -msse2 -msse3 -msse4.1 -msse4.2 -mavx -mavx2 -mfma -mf16c -mprefetchwt1 -DSD_F16C=true -DF_AVX2=true")
check_cxx_compiler_flag("-mno-avx256-split-unaligned-load -mno-avx256-split-unaligned-store" NO_AVX256_SPLIT)
if(NO_AVX256_SPLIT)
set(ARCH_TUNE "${ARCH_TUNE} -mno-avx256-split-unaligned-load -mno-avx256-split-unaligned-store")
endif()
elseif(SD_EXTENSION MATCHES "avx512") # SD_EXTENSION needs to be defined
message("Building AVX512 binary...")
# Note: AVX512 flags were added directly to CMAKE_CXX_FLAGS in blas, which is less ideal.
# Let's add them conditionally here based on compiler.
set(AVX512_FLAGS "-mmmx -msse -msse2 -msse3 -msse4.1 -msse4.2 -mavx -mavx2 -mfma -mf16c -mavx512f -mavx512vl -mavx512bw -mavx512dq -mavx512cd -mbmi -mbmi2 -mprefetchwt1 -mclflushopt -mxsavec -mxsaves -DSD_F16C=true -DF_AVX512=true")
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU|Intel")
set(ARCH_TUNE "${ARCH_TUNE} ${AVX512_FLAGS}")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
# MSVC uses /arch:AVX512 - add this if MSVC and AVX512 extension
set(ARCH_TUNE "${ARCH_TUNE} /arch:AVX512")
else()
message(WARNING "AVX512 extension requested, but compiler ${CMAKE_CXX_COMPILER_ID} might not support the specified flags.")
endif()
else() # Generic x86-64 or other non-tuned arch
if("${SD_ARCH}" STREQUAL "x86-64")
message("Building generic x86_64 binary...")
set(ARCH_TYPE "generic")
add_compile_definitions(F_X64=true)
if(NOT WIN32 AND NOT SD_CUDA AND (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU"))
set(ARCH_TUNE "-march=x86-64 -mtune=generic") # Apply generic tuning only if needed
endif()
else() # Other architectures not specifically handled above
if(NOT WIN32 AND NOT SD_CUDA AND (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU"))
# Apply generic tuning based on SD_ARCH if not Windows/CUDA/GNU
set(ARCH_TUNE "-march=${SD_ARCH} -mtune=${SD_ARCH}")
endif()
endif()
endif()

# Apply ARCH_TUNE based on compiler ID (from blas)
if (NOT "${ARCH_TUNE}" STREQUAL "")
if(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang" AND SD_X86_BUILD)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ARCH_TUNE}")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ARCH_TUNE}")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
# Optimization level already handled in build type flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ARCH_TUNE} -fp-model fast")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
# MSVC tuning often done via /arch:, which might be part of ARCH_TUNE already
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ARCH_TUNE}")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT CMAKE_SYSTEM_NAME MATCHES "Aurora" AND NOT SD_CUDA)
# INFORMATIVE_FLAGS is not defined, omitting. Standard already set.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ARCH_TUNE}")
endif()
endif()


# External Include Directories (from blas)
set(EXTERNAL_INCLUDE_DIRS "")
if(CMAKE_SYSTEM_NAME MATCHES "Linux")
list(APPEND EXTERNAL_INCLUDE_DIRS "/usr/include" "/usr/local/include")
endif()

# Ensure SD_CPU/SD_CUDA are correctly set (from blas)
if(NOT DEFINED SD_CUDA)
set(SD_CUDA FALSE)
endif()
if(NOT DEFINED SD_CPU)
if(NOT SD_CUDA)
set(SD_CPU TRUE)
else()
set(SD_CPU FALSE)
endif()
endif()
if(SD_CUDA)
set(SD_CPU FALSE) # Ensure CPU is false if CUDA is true
else()
set(SD_CPU TRUE) # Ensure CPU is true if CUDA is false
endif()
message("Build Target: CUDA=${SD_CUDA}, CPU=${SD_CPU}")


# Set Library Name (from blas)
if(NOT DEFINED SD_LIBRARY_NAME)
if(SD_CUDA)
set(SD_LIBRARY_NAME nd4jcuda)
else()
set(SD_LIBRARY_NAME nd4jcpu)
endif()
endif()
message("Output library name: ${SD_LIBRARY_NAME}")


# --- Dependency Setup (Mainly from main CMakeLists.txt) ---

# Helper flags for dependencies (can be set via cmake command line)
option(HELPERS_onednn "Build with oneDNN" OFF)
option(HELPERS_armcompute "Build with ARM Compute Library" OFF)
option(HELPERS_cudnn "Build with cuDNN" OFF)

# DNNL (oneDNN) Setup [cite: 104, 105, 106]
set(ONEDNN_LIBRARIES "")
set(HAVE_ONEDNN 0)
if (${HELPERS_onednn})
message("Configuring oneDNN...")
set(HAVE_ONEDNN 1)
add_definitions(-DDNNL_ENABLE_CONCURRENT_EXEC=true) # Definition from main [cite: 90]
set(DNNL_LIBRARY_TYPE "STATIC" CACHE STRING "Hack to enforce static mode" FORCE) [cite: 104]
set(DNNL_CPU_RUNTIME "OMP" CACHE STRING "Set CPU runtime for oneDNN" FORCE) # Default to OpenMP
set(DNNL_GPU_RUNTIME "NONE" CACHE STRING "Disable GPU runtime for oneDNN by default" FORCE) # Disable GPU unless specifically needed

# Download and build oneDNN
if(NOT EXISTS "${CMAKE_CURRENT_BINARY_DIR}/onednn-download-complete.marker")
configure_file(./cmake/CMakeLists.txt.onednn.in onednn-download/CMakeLists.txt @ONLY) # Path adjusted
execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
RESULT_VARIABLE result
WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/onednn-download )
if(result)
message(FATAL_ERROR "CMake configure step for onednn failed: ${result}")
endif()
execute_process(COMMAND ${CMAKE_COMMAND} --build .
RESULT_VARIABLE result
WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/onednn-download ) [cite: 105]
if(result)
message(FATAL_ERROR "Build step for onednn failed: ${result}")
endif()
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/onednn-download-complete.marker" "Download complete")
else()
message(STATUS "oneDNN already downloaded, skipping download step.")
endif()

# Add oneDNN as subdirectory
add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/onednn-src
${CMAKE_CURRENT_BINARY_DIR}/onednn-build
EXCLUDE_FROM_ALL)

set(ONEDNN_INCLUDE_DIRS ${CMAKE_CURRENT_BINARY_DIR}/onednn-build/include ${CMAKE_CURRENT_BINARY_DIR}/onednn-src/include)
include_directories(${ONEDNN_INCLUDE_DIRS})
set(ONEDNN_LIBRARIES dnnl) # Link target name [cite: 106]
add_definitions(-DHAVE_ONEDNN=1) # Ensure definition is set
else()
message("oneDNN helper is OFF. Skipping oneDNN build.")
endif()


# ARM Compute Library Setup [cite: 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103]
set(ARMCOMPUTE_LIBRARIES "")
set(HAVE_ARMCOMPUTE 0)
if(${HELPERS_armcompute})
message(STATUS "ARM Compute helper is active, attempting configuration...")
find_package(ARMCOMPUTE QUIET) # Try system first

if(ARMCOMPUTE_FOUND)
set(HAVE_ARMCOMPUTE 1)
include_directories(${ARMCOMPUTE_INCLUDE_DIRS})
set(ARMCOMPUTE_LIBRARIES ${ARMCOMPUTE_LIBRARIES}) # Should be ARMCOMPUTE_LIBRARIES from find_package
message(STATUS "Found ARM Compute via find_package. Include: ${ARMCOMPUTE_INCLUDE_DIRS}, Libs: ${ARMCOMPUTE_LIBRARIES}") [cite: 93, 94]
add_definitions(-DHAVE_ARMCOMPUTE=1)
add_definitions(-DARMCOMPUTENEON_ENABLED) # Add definition needed by code
else()
message(STATUS "ARM Compute not found via find_package. Checking download conditions.")
if(DEFINED ARM_COMPUTE_URL AND (${SD_ARCH} MATCHES "armv7|armv8-a|arm64"))
message(STATUS "Attempting to download ARM Compute Library from: ${ARM_COMPUTE_URL}")
if(NOT EXISTS "${CMAKE_CURRENT_BINARY_DIR}/armcompute-download-complete.marker")
configure_file(./cmake/CMakeLists.txt.armcompute.in armcompute-download/CMakeLists.txt @ONLY) # Path adjusted [cite: 95]
execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
RESULT_VARIABLE result_config
WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/armcompute-download
OUTPUT_QUIET ERROR_QUIET) [cite: 96]
if(NOT result_config EQUAL 0)
message(WARNING "CMake configuration step for ARM Compute Library download failed: ${result_config}")
else()
execute_process(COMMAND ${CMAKE_COMMAND} --build .
RESULT_VARIABLE result_build
WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/armcompute-download
OUTPUT_QUIET ERROR_QUIET) [cite: 97, 98]
if(NOT result_build EQUAL 0)
message(WARNING "Build step for ARM Compute Library download failed: ${result_build}") [cite: 99]
else()
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/armcompute-download-complete.marker" "Download complete")
set(ARMCOMPUTE_ROOT "${CMAKE_CURRENT_BINARY_DIR}/armcompute-src") # Set the root directory [cite: 99]
set(ARMCOMPUTE_LIB_PATH "${ARMCOMPUTE_ROOT}/lib/libarm_compute.so") [cite: 100]
set(ARMCOMPUTE_CORE_LIB_PATH "${ARMCOMPUTE_ROOT}/lib/libarm_compute_core.so") [cite: 100]

if(EXISTS "${ARMCOMPUTE_LIB_PATH}" AND EXISTS "${ARMCOMPUTE_CORE_LIB_PATH}")
add_definitions(-DARMCOMPUTENEON_ENABLED) [cite: 101]
include_directories(${ARMCOMPUTE_ROOT} ${ARMCOMPUTE_ROOT}/include) [cite: 101]
set(ARMCOMPUTE_LIBRARIES ${ARMCOMPUTE_LIB_PATH} ${ARMCOMPUTE_CORE_LIB_PATH}) [cite: 101]
set(HAVE_ARMCOMPUTE 1) [cite: 101]
add_definitions(-DHAVE_ARMCOMPUTE=1) # Ensure definition is set
message(STATUS "ARM Compute Library successfully configured via download at: ${ARMCOMPUTE_ROOT}") [cite: 102]
else()
message(WARNING "ARM Compute download/build step completed, but library files not found at expected location: ${ARMCOMPUTE_ROOT}/lib") [cite: 102]
endif()
endif() # Build success
endif() # Config success
else()
message(STATUS "ARM Compute already downloaded, configuring from local.")
# Need to re-set paths if marker exists but variables aren't set
set(ARMCOMPUTE_ROOT "${CMAKE_CURRENT_BINARY_DIR}/armcompute-src")
set(ARMCOMPUTE_LIB_PATH "${ARMCOMPUTE_ROOT}/lib/libarm_compute.so")
set(ARMCOMPUTE_CORE_LIB_PATH "${ARMCOMPUTE_ROOT}/lib/libarm_compute_core.so")
if(EXISTS "${ARMCOMPUTE_LIB_PATH}" AND EXISTS "${ARMCOMPUTE_CORE_LIB_PATH}")
add_definitions(-DARMCOMPUTENEON_ENABLED)
include_directories(${ARMCOMPUTE_ROOT} ${ARMCOMPUTE_ROOT}/include)
set(ARMCOMPUTE_LIBRARIES ${ARMCOMPUTE_LIB_PATH} ${ARMCOMPUTE_CORE_LIB_PATH})
set(HAVE_ARMCOMPUTE 1)
add_definitions(-DHAVE_ARMCOMPUTE=1)
message(STATUS "ARM Compute Library configured from existing download at: ${ARMCOMPUTE_ROOT}")
else()
message(WARNING "ARM Compute download marker exists, but library files not found at: ${ARMCOMPUTE_ROOT}/lib")
endif()
endif() # Download marker check
else()
message(STATUS "ARM Compute helper active, but library not found and download conditions not met (URL: ${ARM_COMPUTE_URL}, Arch: ${SD_ARCH}).") [cite: 103]
endif() # Download check
endif() # Find package check
else()
message("ARM Compute helper is OFF. Skipping ARM Compute configuration.")
endif()

# BLAS Setup (CPU Only) (from main) [cite: 90, 91, 92]
set(BLAS_LIBRARIES "")
set(OPENBLAS_LIBRARIES "")
if(SD_CPU)
if ("${OPENBLAS_PATH}" STREQUAL "")
if (NOT APPLE) # Don't use OpenBLAS by default on Apple [cite: 90]
# Try finding OpenBLAS first via CMake's FindBLAS mechanism
set(BLA_VENDOR "OpenBLAS" CACHE STRING "Preferred BLAS vendor")
find_package(BLAS QUIET)
if(BLAS_FOUND AND BLAS_INFO STREQUAL "open")
message("Found OpenBLAS via find_package: ${BLAS_LIBRARIES}")
add_definitions(-D__EXTERNAL_BLAS__=true)
set(HAVE_OPENBLAS 1) # Indicate OpenBLAS is used
else()
message("OpenBLAS not found via find_package BLA_VENDOR. Trying generic BLAS.")
# Reset vendor preference and search again for any BLAS
unset(BLA_VENDOR CACHE)
find_package(BLAS REQUIRED) # Require some BLAS [cite: 91]
if (BLAS_FOUND)
message("Found generic external BLAS implementation: ${BLAS_LIBRARIES} (Vendor: ${BLAS_INFO})")
add_definitions(-D__EXTERNAL_BLAS__=true)
else()
message(FATAL_ERROR "No BLAS library found. Please install OpenBLAS or another BLAS implementation (MKL, ATLAS, Accelerate).")
endif()
endif()
else() # On Apple, prefer Accelerate
set(BLA_VENDOR "Accelerate" CACHE STRING "Preferred BLAS vendor on Apple")
find_package(BLAS REQUIRED)
if (BLAS_FOUND AND BLAS_INFO STREQUAL "accelerate")
message("Found Accelerate BLAS on Apple: ${BLAS_LIBRARIES}")
add_definitions(-D__EXTERNAL_BLAS__=true)
# No specific HAVE_OPENBLAS flag needed for Accelerate
else()
message(WARNING "Could not find Accelerate framework on Apple. Build might fail or use a fallback BLAS if available.")
# Allow build to continue, might pick up another BLAS if installed.
endif()
endif()
else()
# Use externally provided OPENBLAS_PATH [cite: 92]
message("Using OpenBLAS from external path: ${OPENBLAS_PATH}")
set(HAVE_OPENBLAS 1)
include_directories(${OPENBLAS_PATH}/include/)
link_directories(${OPENBLAS_PATH} ${OPENBLAS_PATH}/lib/)
set(OPENBLAS_LIBRARIES openblas) # Library name might vary, assume 'openblas' [cite: 92]
add_definitions(-DHAVE_OPENBLAS=1)
add_definitions(-D__EXTERNAL_BLAS__=true)
endif()
endif()

# CUDA Setup (if SD_CUDA is TRUE) - Integrated from main and blas
set(CUDA_LIBRARIES "")
set(CUDA_CUBLAS_LIBRARIES "")
set(CUDA_cusolver_LIBRARY "")
set(CUDA_DRIVER_LIBRARY "")
set(CUDNN "")
set(HAVE_CUDNN false)

if(SD_CUDA)
message("Building with CUDA support")
add_definitions(-D__CUDABLAS__=true)

# Set flags to allow unsupported compilers *before* enabling CUDA (from main) [cite: 11, 12]
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --allow-unsupported-compiler")
if(WIN32)
message("Setting up Windows-specific CUDA unsupported compiler flags") [cite: 11]
set(ENV{CUDAFE_FLAGS} "--allow-unsupported-compiler")
set(ENV{CUDA_NVCC_FLAGS} "--allow-unsupported-compiler")
else()
message("Setting up Linux/macOS-specific CUDA unsupported compiler flags") [cite: 12]
set(ENV{CUDAFE_FLAGS} "--allow-unsupported-compiler")
set(ENV{NVCC_OPTIONS} "--allow-unsupported-compiler")
endif()
message("Initial CUDA flags set to allow unsupported compiler: ${CMAKE_CUDA_FLAGS}")

# Enable CUDA language
enable_language(CUDA)
find_package(CUDA REQUIRED)

if(CUDA_FOUND)
message("CUDA Found!") [cite: 13]
message("CUDA Version: ${CUDA_VERSION}")
message("CUDA Include Dirs: ${CUDA_INCLUDE_DIRS}") [cite: 13]
include_directories(${CUDA_INCLUDE_DIRS})
set(CUDA_LIBRARIES ${CUDA_LIBRARIES}) # Capture standard CUDA libraries (cudart etc)
set(CUDA_CUBLAS_LIBRARIES ${CUDA_CUBLAS_LIBRARIES}) # Capture standard cublas
set(CUDA_cusolver_LIBRARY ${CUDA_cusolver_LIBRARY}) # Capture standard cusolver

# Set CUDA C++ Standard (from main) [cite: 26]
set(CMAKE_CUDA_STANDARD ${CMAKE_CUDA_STANDARD}) # Use the value set earlier
message("Set CUDA C++ Standard to ${CMAKE_CUDA_STANDARD}") [cite: 26]

# Additional CUDA Flags
set(CMAKE_CUDA_FLAGS_DEBUG " -g") [cite: 13]
if("${SD_PTXAS}" STREQUAL "ON")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v") [cite: 15]
endif()
if(SD_KEEP_NVCC_OUTPUT)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --keep") [cite: 14]
endif()

# Compiler specific flags for CUDA (from main and blas)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
if(SD_GCC_FUNCTRACE STREQUAL "ON")
# Apply functrace flags to CXX flags (already done above)
# Add CUDA specific functrace flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-fPIC --device-debug -lineinfo -G") [cite: 15]
add_compile_definitions(SD_GCC_FUNCTRACE) # Ensure definition exists [cite: 15]
else()
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-fPIC") [cite: 15]
endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
# MSVC specific flags from blas - /EHsc /bigobj /std:c++14 applied above in WIN32 block
# No extra CMAKE_CUDA_FLAGS needed here besides allow-unsupported-compiler
message("Using allow-unsupported-compiler flag for Windows CUDA build")
endif()

# Jetson Nano Specific Library Paths (from blas) [cite: 17, 18, 19]
if("${SD_ARCH}" MATCHES "armv8-a|arm64" AND UNIX AND NOT ANDROID) # More robust check for Jetson-like ARM
if(EXISTS "$ENV{loc_DIR}/cuda/targets/aarch64-linux/lib/stubs/libcublas.so") # Check if path likely exists
message("Applying Jetson Nano specific settings") [cite: 17]
# Check if libraries were found automatically, override if not or pointing to stubs
if(NOT CUDA_CUBLAS_LIBRARIES OR "${CUDA_CUBLAS_LIBRARIES}" MATCHES ".*NOTFOUND.*" OR "${CUDA_CUBLAS_LIBRARIES}" MATCHES ".*stubs.*")
message("Setting cublas library manually for Jetson Nano") [cite: 17]
set(CUDA_CUBLAS_LIBRARIES "$ENV{loc_DIR}/cuda/targets/aarch64-linux/lib/stubs/libcublas.so") # Adjusted variable name
# Unset NOTFOUND cache variables if they exist
unset(CUDA_cublas_LIBRARY-NOTFOUND CACHE)
unset(CUDA_CUBLAS_LIBRARIES-NOTFOUND CACHE)
endif()
if(NOT CUDA_cusolver_LIBRARY OR "${CUDA_cusolver_LIBRARY}" MATCHES ".*NOTFOUND.*" OR "${CUDA_cusolver_LIBRARY}" MATCHES ".*stubs.*")
message("Setting cusolver library manually for Jetson Nano") [cite: 18]
set(CUDA_cusolver_LIBRARY "$ENV{loc_DIR}/cuda/targets/aarch64-linux/lib/stubs/libcusolver.so") [cite: 19]
unset(CUDA_cusolver_LIBRARY-NOTFOUND CACHE)
endif()
message("Jetson Nano cublas library: ${CUDA_CUBLAS_LIBRARIES}, cusolver library: ${CUDA_cusolver_LIBRARY}") [cite: 19]
endif()
endif()

# Windows Library Paths (if not found automatically) (from blas) [cite: 16]
if(WIN32)
if(NOT CUDA_CUBLAS_LIBRARIES AND EXISTS "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/cublas.lib")
message("Setting Windows cublas library manually")
set(CUDA_CUBLAS_LIBRARIES "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/cublas.lib")
endif()
if(NOT CUDA_cusolver_LIBRARY AND EXISTS "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/cusolver.lib")
message("Setting Windows cusolver library manually")
set(CUDA_cusolver_LIBRARY "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/cusolver.lib")
endif()
endif()

# Find CUDA Driver Library (from main) [cite: 25]
find_library(CUDA_DRIVER_LIBRARY
NAMES cuda
PATHS ${CUDA_TOOLKIT_ROOT_DIR} ENV CUDA_HOME
PATH_SUFFIXES lib64 lib lib/x64 lib/Win32 lib/x86_64-linux-gnu) [cite: 25] # Added more suffixes
if(NOT CUDA_DRIVER_LIBRARY)
message(WARNING "CUDA Driver Library (libcuda.so / cuda.lib) not found.")
else()
message("CUDA Driver Library found: ${CUDA_DRIVER_LIBRARY}")
endif()

# Max Register Count (from blas) [cite: 20]
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -maxrregcount=128")

# CUDA Architectures (from blas) [cite: 20, 21, 22, 23, 24]
set(COMPUTE "auto") # Default to auto-detect, can be overridden via cmake -DCOMPUTE=...
string(TOLOWER "${COMPUTE}" COMPUTE_CMP)
set(CUDA_ARCH_FLAGS "")

if(COMPUTE_CMP STREQUAL "all")
# Explicitly list common architectures
if(CUDA_VERSION VERSION_LESS 11.0)
set(CUDA_ARCH_LIST 35 37 50 52 60 61 70 75)
elseif(CUDA_VERSION VERSION_LESS 11.1) # Pascal, Volta, Turing
set(CUDA_ARCH_LIST 60 61 70 75)
elseif(CUDA_VERSION VERSION_LESS 11.8) # Ampere added
set(CUDA_ARCH_LIST 60 61 70 75 80 86)
elseif(CUDA_VERSION VERSION_LESS 12.0) # Hopper pre-support
set(CUDA_ARCH_LIST 60 61 70 75 80 86 89)
else() # Hopper and later (Assume 90 for now)
set(CUDA_ARCH_LIST 60 61 70 75 80 86 89 90)
endif()
message(STATUS "Building for all specified architectures: ${CUDA_ARCH_LIST}")
foreach(ARCH ${CUDA_ARCH_LIST})
list(APPEND CUDA_ARCH_FLAGS "-gencode" "arch=compute_${ARCH},code=sm_${ARCH}")
endforeach()
set(CMAKE_CUDA_ARCHITECTURES "${CUDA_ARCH_LIST}") # Set the list directly
elseif(COMPUTE_CMP STREQUAL "auto")
message(STATUS "Auto-detecting CUDA architecture.")
CUDA_SELECT_NVCC_ARCH_FLAGS(CUDA_ARCH_FLAGS Auto) # Let CMake handle auto-detection
# CMAKE_CUDA_ARCHITECTURES should be set automatically by CUDA_SELECT_NVCC_ARCH_FLAGS
elseif(COMPUTE_CMP MATCHES "^([0-9]+\\.?[0-9]*)(;[0-9]+\\.?[0-9]*)*$") # Matches one or more compute capabilities like 75 or 7.5;8.6
string(REPLACE "." "" COMPUTE_LIST_CLEAN ${COMPUTE_CMP}) # Remove dots for sm_XX format
string(REPLACE ";" " " COMPUTE_LIST_SPACE ${COMPUTE_LIST_CLEAN}) # Space separated
set(CUDA_ARCH_LIST "")
foreach(ARCH ${COMPUTE_LIST_SPACE})
list(APPEND CUDA_ARCH_FLAGS "-gencode" "arch=compute_${ARCH},code=sm_${ARCH}")
list(APPEND CUDA_ARCH_LIST ${ARCH})
endforeach()
set(CMAKE_CUDA_ARCHITECTURES "${CUDA_ARCH_LIST}") # Set the list directly
message(STATUS "Building for specified CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
else()
message(WARNING "Invalid COMPUTE value '${COMPUTE}'. Using default or auto-detected architectures.")
CUDA_SELECT_NVCC_ARCH_FLAGS(CUDA_ARCH_FLAGS Auto) # Fallback to auto
endif()

# Final CUDA Flags Consolidation
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DCUDA_VERSION_MAJOR=${CUDA_VERSION_MAJOR} -w --cudart=shared --expt-extended-lambda -Xfatbin -compress-all ${CUDA_ARCH_FLAGS}") # Added -w from blas
message("Final CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")

# cuDNN Setup (if HELPERS_cudnn is TRUE) (from main) [cite: 107, 108, 109, 110, 111]
if(${HELPERS_cudnn})
message("Configuring cuDNN...")
SET(CUDNN_LIBNAME "cudnn") [cite: 107]

# Environment variable hints
if(DEFINED ENV{CUDNN_ROOT_DIR}) set(CUDNN_ROOT_DIR $ENV{CUDNN_ROOT_DIR}) [cite: 107]
if(DEFINED ENV{CUDA_TOOLKIT_ROOT_DIR}) set(CUDA_TOOLKIT_ROOT_DIR $ENV{CUDA_TOOLKIT_ROOT_DIR}) [cite: 107]
message("CUDNN Hints: CUDNN_ROOT_DIR=${CUDNN_ROOT_DIR}, CUDA_TOOLKIT_ROOT_DIR=${CUDA_TOOLKIT_ROOT_DIR}") [cite: 107]

find_path(CUDNN_INCLUDE_DIR cudnn.h
HINTS ${CUDNN_ROOT_DIR} ${CUDA_TOOLKIT_ROOT_DIR} ENV CUDNN_ROOT_DIR ENV CUDA_TOOLKIT_ROOT_DIR
PATH_SUFFIXES cuda/include include include/cuda) [cite: 108]

find_library(CUDNN_LIBRARY ${CUDNN_LIBNAME}
HINTS ${CUDNN_ROOT_DIR} ${CUDA_TOOLKIT_ROOT_DIR} ENV CUDNN_ROOT_DIR ENV CUDA_TOOLKIT_ROOT_DIR
PATH_SUFFIXES lib lib64 cuda/lib cuda/lib64 lib/x64 lib/x86_64-linux-gnu) [cite: 109] # Added common suffix

message("CUDNN_INCLUDE_DIR search result: ${CUDNN_INCLUDE_DIR}") [cite: 109]
message("CUDNN_LIBRARY search result: ${CUDNN_LIBRARY}") [cite: 109]

if(CUDNN_LIBRARY AND CUDNN_INCLUDE_DIR)
message("Found cuDNN: include at ${CUDNN_INCLUDE_DIR}, library at ${CUDNN_LIBRARY}") [cite: 110]
include_directories(${CUDNN_INCLUDE_DIR})
set(HAVE_CUDNN true)
add_definitions(-DHAVE_CUDNN=1)
set(CUDNN ${CUDNN_LIBRARY}) [cite: 110]
else()
message(WARNING "cuDNN not found. Build will continue without cuDNN support.") [cite: 111]
set(HAVE_CUDNN false)
endif()
else()
message("cuDNN helper is OFF. Skipping cuDNN configuration.")
endif() # HELPERS_cudnn

set(DEFAULT_ENGINE "samediff::ENGINE_CUDA") [cite: 26]
else()
message(FATAL_ERROR "CUDA requested but not found. Please ensure CUDA toolkit is installed and configured correctly.")
set(SD_CUDA FALSE) # Ensure flag is false if not found
set(SD_CPU TRUE) # Fallback to CPU
endif() # CUDA_FOUND
else() # Not SD_CUDA
message("Building with CPU support only.")
set(SD_CPU TRUE) # Explicitly set CPU true
set(DEFAULT_ENGINE "samediff::ENGINE_CPU") [cite: 46]
# Ensure HAVE_CUDNN is false if not building with CUDA
set(HAVE_CUDNN false)
endif() # SD_CUDA

# Flatbuffers Setup (from main) [cite: 111, 112, 113, 114, 115, 116]
# Determine if generation is needed
set(GENERATE_FLATC OFF)
if(DEFINED ENV{GENERATE_FLATC} OR DEFINED GENERATE_FLATC_OPTION) # Use a standard CMake option
set(GENERATE_FLATC ON)
endif()
set(FLATBUFFERS_BUILD_FLATC ${GENERATE_FLATC} CACHE BOOL "Build the flatbuffers compiler (flatc)") [cite: 111] # Use the derived flag

# Download flatbuffers
if(NOT EXISTS "${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-download-complete.marker")
message("Downloading flatbuffers...")
configure_file(cmake/CMakeLists.txt.flatbuffers.in flatbuffers-download/CMakeLists.txt @ONLY) # Path adjusted
execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
RESULT_VARIABLE result
WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-download)
if(result)
message(FATAL_ERROR "CMake step for flatbuffers failed: ${result}") [cite: 112]
endif()
execute_process(COMMAND ${CMAKE_COMMAND} --build .
RESULT_VARIABLE result
WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-download)
if(result)
message(FATAL_ERROR "Build step for flatbuffers failed: ${result}") [cite: 112]
endif()
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-download-complete.marker" "Download complete")
else()
message(STATUS "Flatbuffers already downloaded, skipping download step.") [cite: 112]
endif()

# Add flatbuffers subdirectory
add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-src
${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-build
EXCLUDE_FROM_ALL) [cite: 113]
include_directories(${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-src/include) # Include flatbuffers headers [cite: 116]

# Generate Flatbuffers sources if requested [cite: 113, 114, 115, 116]
if(GENERATE_FLATC)
message("Generating flatbuffers sources...")
# Ensure flatc is built first
add_custom_target(build_flatc ALL
COMMAND ${CMAKE_COMMAND} --build ${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-build --target flatc
COMMENT "Building flatc executable"
)
set(FLATC_EXECUTABLE "${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-build/flatc")
message("Using flatc from: ${FLATC_EXECUTABLE}") [cite: 114]

# Define generation target
add_custom_command(
OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/include/graph/generated/sd/graph_generated.h # Example output file
COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_SOURCE_DIR}/include/graph/generated/sd
COMMAND ${CMAKE_COMMAND} -E env FLATC_PATH=${FLATC_EXECUTABLE} bash ${CMAKE_CURRENT_SOURCE_DIR}/flatc-generate.sh # Assumes script exists [cite: 114]
COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/copy-flatc-java.sh # Assumes script exists [cite: 115]
WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
DEPENDS build_flatc ${CMAKE_CURRENT_SOURCE_DIR}/flatc-generate.sh ${CMAKE_CURRENT_SOURCE_DIR}/copy-flatc-java.sh # Depend on flatc build and scripts
COMMENT "Generating flatbuffers C++ and Java sources"
VERBATIM
)
add_custom_target(generate_flatbuffers ALL DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/include/graph/generated/sd/graph_generated.h) # Target depends on output
else()
message("Skipping flatbuffers source generation (GENERATE_FLATC is OFF).")
endif()


# --- Compile Definitions (Ops and Types - from blas) ---
set(DEFINITIONS_CONTENT "")
if(SD_ALL_OPS OR "${SD_OPS_LIST}" STREQUAL "")
message("Adding all ops due to empty op list or SD_ALL_OPS definition: SD_ALL_OPS=${SD_ALL_OPS} SD_OPS_LIST=${SD_OPS_LIST}")
add_compile_definitions(SD_ALL_OPS=1)
string(APPEND DEFINITIONS_CONTENT "#define SD_ALL_OPS 1\n")
else()
message("_OPS: ${SD_OPS_LIST}")
foreach(OP ${SD_OPS_LIST})
add_compile_definitions(OP_${OP}=1)
message(STATUS "OP: ${OP}")
string(APPEND DEFINITIONS_CONTENT "#define OP_${OP} 1\n")
endforeach()
endif()

list(LENGTH SD_TYPES_LIST SD_TYPES_LIST_COUNT)
if(SD_TYPES_LIST_COUNT GREATER 0)
message("Using selective types: ${SD_TYPES_LIST}")
add_compile_definitions(SD_SELECTIVE_TYPES)
string(APPEND DEFINITIONS_CONTENT "#define SD_SELECTIVE_TYPES\n")
foreach(SD_TYPE ${SD_TYPES_LIST})
string(TOUPPER ${SD_TYPE} SD_TYPE_UPPERCASE)
add_compile_definitions(HAS_${SD_TYPE_UPPERCASE})
message(STATUS "TYPE: ${SD_TYPE_UPPERCASE}")
string(APPEND DEFINITIONS_CONTENT "#define HAS_${SD_TYPE_UPPERCASE}\n")
endforeach()
else()
message("Using all default types (SD_TYPES_LIST not provided or empty).")
# Define default types if none are provided? Or assume all are enabled?
# Assume code handles lack of HAS_ definitions by enabling all for now.
endif()

# Write Definitions to Header File (from blas)
set(OP_OUTPUT_FILE "${CMAKE_BINARY_DIR}/include/sd_definitions_generated.h") # Define output path relative to build dir
message("Op/Type definitions will be written to \"${OP_OUTPUT_FILE}\"")
file(WRITE "${OP_OUTPUT_FILE}" "#ifndef SD_DEFINITIONS_GEN_H_\n#define SD_DEFINITIONS_GEN_H_\n${DEFINITIONS_CONTENT}\n#endif\n")
include_directories(${CMAKE_BINARY_DIR}/include) # Ensure build dir include path


# --- Configuration Header (from main) ---
configure_file(include/config.h.in include/config.h @ONLY) # Configure to build directory
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)


# --- Helper Functions (from blas and main) ---

# removeFileIfExcluded (from main, with fixes from blas) [cite: 64, 65, 66, 67, 68]
function(removeFileIfExcluded)
cmake_parse_arguments(
PARSED_ARGS # prefix of output variables
""          # list of names of the boolean arguments (only defined ones will be true)
"FILE_ITEM" # list of names of mono-valued arguments
"LIST_ITEM" # list of names of multi-valued arguments (expects list *name*)
${ARGN}     # arguments of the function to parse, here we take the all original ones
) [cite: 64]

# Check if SD_OPS_LIST is defined and not empty
if(DEFINED SD_OPS_LIST AND NOT "${SD_OPS_LIST}" STREQUAL "")
file(READ ${PARSED_ARGS_FILE_ITEM} FILE_CONTENTS)
string(FIND "${FILE_CONTENTS}" "NOT_EXCLUDED" NOT_EXCLUDED_IDX)

# Only process if the marker is present
if(${NOT_EXCLUDED_IDX} GREATER_EQUAL 0) [cite: 65]
set(local_list ${${PARSED_ARGS_LIST_ITEM}}) # Work on a copy [cite: 65]
set(file_should_be_removed TRUE) # Assume removal unless an allowed OP is found

foreach(OP ${SD_OPS_LIST})
string(FIND "${FILE_CONTENTS}" "NOT_EXCLUDED(OP_${OP})" NOT_EXCLUDED_OP_IDX)
# If *any* required OP has an exclusion marker, the file should be kept
if(${NOT_EXCLUDED_OP_IDX} GREATER_EQUAL 0)
set(file_should_be_removed FALSE)
break() # Keep the file, no need to check other OPs
endif()
endforeach()

# If after checking all required OPs, no exclusion marker was found, remove the file
if(file_should_be_removed)
list(REMOVE_ITEM local_list "${PARSED_ARGS_FILE_ITEM}")
set(${PARSED_ARGS_LIST_ITEM} ${local_list} PARENT_SCOPE) # Propagate change back [cite: 67]
message(STATUS "Excluding file (no matching NOT_EXCLUDED OP): ${PARSED_ARGS_FILE_ITEM}")
endif()
endif() # NOT_EXCLUDED marker check
endif() # SD_OPS_LIST check
endfunction() [cite: 68]


# genCompilation (from main/blas) [cite: 61, 62, 63, 69, 70, 71, 72, 73, 74]
function(genCompilation FILE_ITEM OUTPUT_LIST_VAR)
get_filename_component(FILE_ITEM_WE ${FILE_ITEM} NAME_WE) # Use FILE_ITEM directly
set(EXTENSION "cpp")
if(FILE_ITEM MATCHES "\\.cu\\.in$")
set(EXTENSION "cu")
endif()

file(READ ${FILE_ITEM} CONTENT_FL)

# Determine which type sets are requested by the template
set (SD_FLOAT_TYPES_GEN 0)
set (SD_INTEGER_TYPES_GEN 0)
set (SD_COMMON_TYPES_GEN 0)
set (SD_PAIRWISE_TYPES_GEN 0)
set (RANGE_STOP -1)

string(REGEX MATCHALL "#cmakedefine[ \t]+SD_(INTEGER|COMMON|FLOAT|PAIRWISE)_TYPES_GEN" TYPE_MATCHES ${CONTENT_FL}) [cite: 69]

# Define end indices for type lists (adjust if types.h changes)
set(SD_INTEGER_TYPES_END 7) # Example value
set(SD_COMMON_TYPES_END 12)  # Example value
set(SD_FLOAT_TYPES_END 3)   # Example value
set(SD_PAIRWISE_TYPES_END 12) # Example value

foreach(TYPEX ${TYPE_MATCHES})
set(STOP -1)
if(TYPEX MATCHES "SD_INTEGER_TYPES_GEN$")
set (SD_INTEGER_TYPES_GEN 1)
set(STOP ${SD_INTEGER_TYPES_END}) [cite: 69]
elseif(TYPEX MATCHES "SD_COMMON_TYPES_GEN$")
set (SD_COMMON_TYPES_GEN 1)
set(STOP ${SD_COMMON_TYPES_END}) [cite: 70]
elseif(TYPEX MATCHES "SD_FLOAT_TYPES_GEN$")
set (SD_FLOAT_TYPES_GEN 1)
set(STOP ${SD_FLOAT_TYPES_END}) [cite: 70]
elseif(TYPEX MATCHES "SD_PAIRWISE_TYPES_GEN$")
set (SD_PAIRWISE_TYPES_GEN 1)
set(STOP ${SD_PAIRWISE_TYPES_END}) [cite: 71]
endif()
if(STOP GREATER RANGE_STOP)
set(RANGE_STOP ${STOP}) [cite: 71]
endif()
endforeach()

set(generated_files_list "") # Local list for this function call

if(RANGE_STOP GREATER -1)
message(STATUS "Generating compilation units for ${FILE_ITEM_WE} up to index ${RANGE_STOP}")
# Create output directory
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/generated_compilation_units")

foreach(FL_TYPE_INDEX RANGE 0 ${RANGE_STOP})
# Temporarily set flags for configure_file based on current index
set (CURRENT_SD_FLOAT_TYPES_GEN ${SD_FLOAT_TYPES_GEN})
set (CURRENT_SD_INTEGER_TYPES_GEN ${SD_INTEGER_TYPES_GEN})
set (CURRENT_SD_COMMON_TYPES_GEN ${SD_COMMON_TYPES_GEN})
set (CURRENT_SD_PAIRWISE_TYPES_GEN ${SD_PAIRWISE_TYPES_GEN})

if(FL_TYPE_INDEX GREATER ${SD_FLOAT_TYPES_END}) set (CURRENT_SD_FLOAT_TYPES_GEN 0) [cite: 72]
if(FL_TYPE_INDEX GREATER ${SD_INTEGER_TYPES_END}) set (CURRENT_SD_INTEGER_TYPES_GEN 0) [cite: 72]
if(FL_TYPE_INDEX GREATER ${SD_COMMON_TYPES_END}) set (CURRENT_SD_COMMON_TYPES_GEN 0) [cite: 73]
# No PAIRWISE check needed here? Assuming it uses same range as COMMON?

# Check if *any* type set is active for this index
if(CURRENT_SD_FLOAT_TYPES_GEN OR CURRENT_SD_INTEGER_TYPES_GEN OR CURRENT_SD_COMMON_TYPES_GEN OR CURRENT_SD_PAIRWISE_TYPES_GEN)
set(GENERATED_SOURCE "${CMAKE_BINARY_DIR}/generated_compilation_units/${FILE_ITEM_WE}_${FL_TYPE_INDEX}.${EXTENSION}")
# Pass the CURRENT_ flags to configure_file via Defines
configure_file("${FILE_ITEM}" "${GENERATED_SOURCE}" @ONLY
DEFINES
SD_FLOAT_TYPES_GEN=${CURRENT_SD_FLOAT_TYPES_GEN}
SD_INTEGER_TYPES_GEN=${CURRENT_SD_INTEGER_TYPES_GEN}
SD_COMMON_TYPES_GEN=${CURRENT_SD_COMMON_TYPES_GEN}
SD_PAIRWISE_TYPES_GEN=${CURRENT_SD_PAIRWISE_TYPES_GEN}
FL_TYPE_INDEX=${FL_TYPE_INDEX} # Ensure index is passed
)
list(APPEND generated_files_list ${GENERATED_SOURCE})
endif()
endforeach()
else()
message(STATUS "No type generation directives found in ${FILE_ITEM_WE}")
endif()

# Append generated files to the output list variable in the parent scope
set(${OUTPUT_LIST_VAR} ${${OUTPUT_LIST_VAR}} ${generated_files_list} PARENT_SCOPE) [cite: 74]
endfunction()


# genSingleFunctionCuda (from main/blas) [cite: 47, 48, 49, 50, 51, 52, 75, 76]
function(genSingleFunctionCuda TEMPLATE_FILE COMBINATION OUTPUT_DIR)
string(REPLACE "," ";" COMB_LIST "${COMBINATION}")
list(GET COMB_LIST 0 COMB1)
list(GET COMB_LIST 1 COMB2)
list(GET COMB_LIST 2 COMB3)

get_filename_component(TEMPLATE_BASE "${TEMPLATE_FILE}" NAME_WE) [cite: 47, 75]
file(READ "${TEMPLATE_FILE}" TEMPLATE_CONTENT) [cite: 75]

# Extract class, method, and args - more robust regex needed
# Example regex (adjust based on actual template format):
# Assuming format like: // Function: some::namespace::Class::methodName(...)
# Or template <...> void some::namespace::Class::methodName(...)
string(REGEX MATCH "template[ \n\t<].*[ \n\t>]([a-zA-Z0-9_:]+)::([a-zA-Z0-9_]+)<.*>\\(([^)]*)\\);" FUNC_MATCH "${TEMPLATE_CONTENT}")
if(NOT CMAKE_MATCH_COUNT EQUAL 3)
# Fallback or alternative regex
string(REGEX MATCH "//[ \t]*Function:[ \t]*([a-zA-Z0-9_:]+)::([a-zA-Z0-9_]+)\\(([^)]*)\\)" FUNC_MATCH "${TEMPLATE_CONTENT}")
endif()

if(CMAKE_MATCH_COUNT EQUAL 3)
set(CLASS_NAME ${CMAKE_MATCH_1})
set(METHOD_NAME ${CMAKE_MATCH_2})
set(FUNCTION_ARGS_STRIP ${CMAKE_MATCH_3}) # Args without surrounding ()
string(REGEX REPLACE "::" "_" CLASS_NAME_CLEAN "${CLASS_NAME}") [cite: 75]

# Create a signature hash for uniqueness (from main) [cite: 48, 49, 50]
set(SIGNATURE_ID "")
string(REPLACE "," ";" ARGS_LIST "${FUNCTION_ARGS_STRIP}")
foreach(ARG ${ARGS_LIST})
string(STRIP "${ARG}" ARG_STRIPPED)
# Extract type - needs careful regex
string(REGEX MATCH "^([^ ]+[ ]?[\*&]?)" TYPE_NAME "${ARG_STRIPPED}") # Basic type match
if(CMAKE_MATCH_1)
string(APPEND SIGNATURE_ID "_${CMAKE_MATCH_1}")
endif()
endforeach()
string(REGEX REPLACE "[*& ]" "" SIGNATURE_ID "${SIGNATURE_ID}") # Clean signature
if(SIGNATURE_ID MATCHES ".{30,}") # Hash if too long [cite: 49]
string(MD5 SIGNATURE_HASH "${SIGNATURE_ID}")
string(SUBSTRING "${SIGNATURE_HASH}" 0 8 SIGNATURE_ID)
set(SIGNATURE_ID "_h${SIGNATURE_ID}") [cite: 49]
endif()

# Output filename with signature hash
set(OUTPUT_FILE "${CLASS_NAME_CLEAN}_${METHOD_NAME}${SIGNATURE_ID}_${COMB1}_${COMB2}_${COMB3}.cu") [cite: 50]
set(GENERATED_FILE "${OUTPUT_DIR}/${OUTPUT_FILE}")

# Check existence (from main) [cite: 50]
if(EXISTS "${GENERATED_FILE}")
list(APPEND CUDA_GENERATED_SOURCES "${GENERATED_FILE}")
set(CUDA_GENERATED_SOURCES ${CUDA_GENERATED_SOURCES} PARENT_SCOPE)
return()
endif()

# Streamlined content generation (from main) [cite: 51]
# Find marker or beginning of template definition
string(FIND "${TEMPLATE_CONTENT}" "template <" START_POS) # Basic marker
if(START_POS EQUAL -1)
# Try finding the comment marker
string(FIND "${TEMPLATE_CONTENT}" "// Function:" START_POS)
endif()
if(START_POS EQUAL -1)
message(WARNING "Could not find template start or // Function marker in ${TEMPLATE_FILE}. Using full header.")
set(HEADER_CONTENT ${TEMPLATE_CONTENT}) # Use all if marker not found
else()
string(SUBSTRING "${TEMPLATE_CONTENT}" 0 ${START_POS} HEADER_CONTENT) # Extract header
endif()

# Find full function signature including template parameters
string(REGEX MATCH "(template[ \n\t<][^>]+>[ \n\t]+[a-zA-Z0-9_:]+::[a-zA-Z0-9_]+<[^>]+>\\([^)]*\\));" FULL_SIGNATURE_MATCH "${TEMPLATE_CONTENT}")
if(NOT CMAKE_MATCH_1)
message(FATAL_ERROR "Could not extract full function signature from ${TEMPLATE_FILE}")
endif()
set(FUNCTION_SIGNATURE_NO_SEMICOLON ${CMAKE_MATCH_1})

# Replace template args with concrete types
string(REPLACE "T" "SD_SINGLE_TYPE_${COMB1}" TEMP_SIG "${FUNCTION_SIGNATURE_NO_SEMICOLON}") # Assume T, X, Y
string(REPLACE "X" "SD_SINGLE_TYPE_${COMB2}" TEMP_SIG "${TEMP_SIG}")
string(REPLACE "Y" "SD_SINGLE_TYPE_${COMB3}" INSTANTIATED_SIGNATURE "${TEMP_SIG}")


# Reconstruct the instantiation line
set(NEW_CONTENT "${HEADER_CONTENT}\n\n// Single function instantiation for ${CLASS_NAME}::${METHOD_NAME} (${COMB1},${COMB2},${COMB3})\n")
string(APPEND NEW_CONTENT "template void ${INSTANTIATED_SIGNATURE};\n") # Explicit instantiation


file(MAKE_DIRECTORY "${OUTPUT_DIR}") [cite: 76]
file(WRITE "${GENERATED_FILE}" "${NEW_CONTENT}") [cite: 76]

set_source_files_properties("${GENERATED_FILE}" PROPERTIES LANGUAGE CUDA) [cite: 52, 76]
list(APPEND CUDA_GENERATED_SOURCES "${GENERATED_FILE}")
set(CUDA_GENERATED_SOURCES ${CUDA_GENERATED_SOURCES} PARENT_SCOPE) [cite: 52, 76]
message(STATUS "Generated CUDA single function: ${GENERATED_FILE}") [cite: 52]
else()
message(FATAL_ERROR "Could not extract Class/Method/Args from CUDA template: ${TEMPLATE_FILE}")
endif()
endfunction()

# genPartitionCombination (from main) [cite: 53, 54, 55, 56, 57] - For CPU combination templates
function(genPartitionCombination TEMPLATE_FILE COMBINATION_TYPE COMBINATION OUTPUT_DIR)
string(REPLACE "," ";" COMB_LIST "${COMBINATION}") [cite: 53]
list(LENGTH COMB_LIST COMB_COUNT)

if(NOT (COMBINATION_TYPE EQUAL 3 OR COMBINATION_TYPE EQUAL 2))
message(FATAL_ERROR "Unsupported COMBINATION_TYPE: ${COMBINATION_TYPE}. Use 3 or 2.") [cite: 54]
endif()
if(NOT ((COMBINATION_TYPE EQUAL 3 AND COMB_COUNT EQUAL 3) OR (COMBINATION_TYPE EQUAL 2 AND COMB_COUNT EQUAL 2)))
message(FATAL_ERROR "Combination length (${COMB_COUNT}) does not match COMBINATION_TYPE (${COMBINATION_TYPE}).") [cite: 54]
endif()

# Read the template content once
get_filename_component(TEMPLATE_NAME ${TEMPLATE_FILE} NAME)
file(READ "${TEMPLATE_FILE}" TEMPLATE_CONTENT)

# Perform placeholder replacements and generate file
set(FINAL_CONTENT "${TEMPLATE_CONTENT}")
set(OUTPUT_FILE_SUFFIX "")
if(COMBINATION_TYPE EQUAL 3)
list(GET COMB_LIST 0 COMB1)
list(GET COMB_LIST 1 COMB2)
list(GET COMB_LIST 2 COMB3)
string(REPLACE "@COMB1@" "${COMB1}" FINAL_CONTENT "${FINAL_CONTENT}") [cite: 55, 56]
string(REPLACE "@COMB2@" "${COMB2}" FINAL_CONTENT "${FINAL_CONTENT}") [cite: 55, 56]
string(REPLACE "@COMB3@" "${COMB3}" FINAL_CONTENT "${FINAL_CONTENT}") [cite: 55, 56]
set(OUTPUT_FILE_SUFFIX "_${COMB1}_${COMB2}_${COMB3}")
elseif(COMBINATION_TYPE EQUAL 2)
list(GET COMB_LIST 0 COMB1)
list(GET COMB_LIST 1 COMB2)
string(REPLACE "@COMB1@" "${COMB1}" FINAL_CONTENT "${FINAL_CONTENT}") [cite: 55, 56]
string(REPLACE "@COMB2@" "${COMB2}" FINAL_CONTENT "${FINAL_CONTENT}") [cite: 55, 56]
set(OUTPUT_FILE_SUFFIX "_${COMB1}_${COMB2}")
endif()

# Define the output file name
get_filename_component(TEMPLATE_BASE ${TEMPLATE_FILE} NAME_WE)
set(OUTPUT_FILE "${TEMPLATE_BASE}${OUTPUT_FILE_SUFFIX}.cpp") [cite: 57] # Use template base name + combo
set(GENERATED_FILE "${OUTPUT_DIR}/${OUTPUT_FILE}")

# Write the processed content
file(WRITE "${GENERATED_FILE}" "${FINAL_CONTENT}")

# Append to the generic sources list (passed implicitly via parent scope)
list(APPEND CUSTOMOPS_GENERIC_SOURCES "${GENERATED_FILE}")
set(CUSTOMOPS_GENERIC_SOURCES ${CUSTOMOPS_GENERIC_SOURCES} PARENT_SCOPE)

message(STATUS "Generated CPU Instantiation File: ${GENERATED_FILE}") [cite: 57]
endfunction()


# --- Main Build Logic (CUDA / CPU) ---

# Source file lists initialization
set(ALL_SOURCES "")
set(CUDA_GENERATED_SOURCES "")
set(CUSTOMOPS_GENERIC_SOURCES "") # Initialize list for generated CPU sources
set(CUSTOMOPS_ONEDNN_SOURCES "")
set(CUSTOMOPS_ARMCOMPUTE_SOURCES "")
set(CUSTOMOPS_CUDNN_SOURCES "")
set(CPU_SPECIFIC_SOURCES "")
set(CUDA_SPECIFIC_SOURCES "")

# Common include directories
include_directories(
${CMAKE_CURRENT_SOURCE_DIR}/include
${CMAKE_CURRENT_BINARY_DIR}/include # For generated headers
${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-src/include # Flatbuffers
)
if(HAVE_ONEDNN)
include_directories(${ONEDNN_INCLUDE_DIRS})
endif()
if(HAVE_ARMCOMPUTE)
include_directories(${ARMCOMPUTE_ROOT} ${ARMCOMPUTE_ROOT}/include) # From main's setup [cite: 101]
endif()
if(HAVE_CUDNN)
include_directories(${CUDNN_INCLUDE_DIR}) [cite: 110]
endif()
if(SD_CPU AND HAVE_OPENBLAS)
include_directories(${OPENBLAS_PATH}/include/) [cite: 92]
endif()
list(APPEND ALL_INCLUDE_DIRS ${EXTERNAL_INCLUDE_DIRS}) # Add OS-specific external includes

# --- CUDA Build Path ---
if(SD_CUDA)
message("--- Configuring CUDA Build ---")

# Include TypeMST for CUDA combination generation
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/TypeMST.cmake) # Path from blas [cite: 25]

# Create output directory for CUDA generated files
set(CUDA_GENERATED_DIR "${CMAKE_BINARY_DIR}/cuda_generated")
file(MAKE_DIRECTORY "${CUDA_GENERATED_DIR}")

# Determine number of types for generation
if(DEFINED SD_TYPES_LIST AND SD_TYPES_LIST_COUNT GREATER 0)
set(NUM_TYPES ${SD_TYPES_LIST_COUNT})
message(STATUS "Using ${NUM_TYPES} types from SD_TYPES_LIST for CUDA generation")
else()
set(NUM_TYPES 10) # Default from blas
message(STATUS "Using default number of types for CUDA generation: ${NUM_TYPES}")
endif()

# Generate basic compilation units (.cu.in -> .cu)
file(GLOB CUDA_COMPILATION_UNITS_IN ../include/loops/cuda/compilation_units/*.cu.in) # Path from blas
foreach(FL_ITEM ${CUDA_COMPILATION_UNITS_IN})
genCompilation(${FL_ITEM} CUDA_GENERATED_SOURCES) # Pass output list var
endforeach()

# Generate combination-based CUDA functions
set(SD_USE_MST_TYPES ON) # Default from blas
if(SD_USE_MST_TYPES)
generate_mst_combinations() # Function from TypeMST.cmake
message(STATUS "Using optimized MST type combinations for CUDA: ${COMBINATIONS_3}")
else()
# Define ALL_TYPE_COMBINATIONS if needed for the 'else' case
# set(COMBINATIONS_3 ${ALL_TYPE_COMBINATIONS}) # Placeholder
message(STATUS "Using all possible type combinations for CUDA (SD_USE_MST_TYPES=OFF)")
endif()

file(GLOB CUDA_TEMPLATE_FILES "../include/loops/cuda/comb_compilation_units/*.cu.in") # Path from blas
foreach(TEMPLATE_FILE ${CUDA_TEMPLATE_FILES})
foreach(COMBINATION ${COMBINATIONS_3}) # Use the generated combinations
genSingleFunctionCuda(${TEMPLATE_FILE} ${COMBINATION} "${CUDA_GENERATED_DIR}")
endforeach()
endforeach()


# Gather CUDA Source Files (Globs from blas CUDA section) [cite: 27, 28, 29, 30, 31, 32]
file(GLOB_RECURSE PERF_SOURCES ../include/performance/*.cpp ../include/performance/*.h)
file(GLOB_RECURSE EXCEPTIONS_SOURCES ../include/exceptions/*.cpp ../include/exceptions/*.h)
file(GLOB_RECURSE EXEC_SOURCES ../include/execution/impl/*.cpp ../include/execution/*.h)
file(GLOB_RECURSE EXEC_SOURCES_CUDA ../include/execution/cuda/*.cu ../include/execution/cuda/*.h ../include/execution/*.cu) # CUDA specific
file(GLOB_RECURSE TYPES_SOURCES ../include/types/*.cpp ../include/types/*.h)
file(GLOB_RECURSE ARRAY_SOURCES ../include/array/impl/*.cpp ../include/array/*.h)
file(GLOB_RECURSE ARRAY_SOURCES_CUDA ../include/array/cuda/*.cu ../include/array/cuda/*.chpp) # CUDA specific
file(GLOB_RECURSE MEMORY_SOURCES ../include/memory/impl/*.cpp ../include/memory/*.h)
file(GLOB_RECURSE MEMORY_SOURCES_CUDA ../include/memory/cuda/*.cu) # CUDA specific
file(GLOB_RECURSE GRAPH_SOURCES ../include/graph/*.cpp ../include/graph/*.h)
file(GLOB_RECURSE GRAPH_SOURCES_CUDA ../include/graph/*.cu) # CUDA specific [cite: 27]
file(GLOB_RECURSE CUSTOMOPS_SOURCES ../include/ops/declarable/generic/*.cpp) # Generic ops implementation
file(GLOB_RECURSE CUSTOMOPS_HELPERS_IMPL ../include/ops/declarable/helpers/impl/*.cpp) # Common helpers impl
file(GLOB_RECURSE CUSTOMOPS_HELPERS_CUDA ../include/ops/declarable/helpers/cuda/*.cu) # CUDA helpers impl [cite: 28]
file(GLOB_RECURSE OPS_SOURCES ../include/ops/impl/*.cpp ../include/ops/declarable/impl/*.cpp ../include/ops/*.h)
file(GLOB_RECURSE HELPERS_SOURCES ../include/build_info.cpp ../include/ConstMessages.cpp ../include/helpers/*.cpp ../include/helpers/*.h)
file(GLOB_RECURSE HELPERS_SOURCES_CUDA ../include/helpers/cuda/*.cu) # CUDA specific helpers [cite: 28]
file(GLOB_RECURSE INDEXING_SOURCES ../include/indexing/*.cpp ../include/indexing/*.h)
file(GLOB_RECURSE LOOPS_SOURCES ../include/loops/impl/*.cpp ../include/loops/*.h)
file(GLOB_RECURSE LOOPS_SOURCES_CUDA ../include/loops/*.cu ../include/loops/cuda/**/*.cu) # CUDA specific loops [cite: 29]
file(GLOB_RECURSE LEGACY_SOURCES ../include/legacy/impl/*.cpp ../include/legacy/*.h)
file(GLOB_RECURSE LEGACY_SOURCES_CUDA ../include/legacy/*.cu) # CUDA specific legacy

# Platform specific sources (ONEDNN, ARMCompute, CUDNN)
if(HAVE_ONEDNN)
file(GLOB_RECURSE CUSTOMOPS_ONEDNN_SOURCES ../include/ops/declarable/platform/mkldnn/*.cpp ../include/ops/declarable/platform/mkldnn/mkldnnUtils.h) [cite: 31]
endif()
if(HAVE_ARMCOMPUTE) # Check if ARM Compute is actually enabled and found
file(GLOB_RECURSE CUSTOMOPS_ARMCOMPUTE_SOURCES ../include/ops/declarable/platform/armcompute/*.cpp ../include/ops/declarable/platform/armcompute/*.h) [cite: 31]
endif()
if(HAVE_CUDNN) # Check if cuDNN is actually enabled and found
file(GLOB_RECURSE CUSTOMOPS_CUDNN_SOURCES ../include/ops/declarable/platform/cudnn/*.cu) [cite: 33]
endif()

# Combine all sources for CUDA build
set(ALL_SOURCES
${CUDA_GENERATED_SOURCES} # Generated .cu files
${PERF_SOURCES}
${EXCEPTIONS_SOURCES} [cite: 30, 36]
${EXEC_SOURCES} ${EXEC_SOURCES_CUDA} [cite: 30, 34, 39]
${TYPES_SOURCES} [cite: 30, 35, 39]
${ARRAY_SOURCES} ${ARRAY_SOURCES_CUDA} [cite: 30, 34, 39]
${MEMORY_SOURCES} ${MEMORY_SOURCES_CUDA} [cite: 30, 35, 39]
${GRAPH_SOURCES} ${GRAPH_SOURCES_CUDA} [cite: 30, 35, 40]
${CUSTOMOPS_SOURCES} [cite: 31, 35, 40]
${CUSTOMOPS_HELPERS_IMPL} ${CUSTOMOPS_HELPERS_CUDA} [cite: 34, 38]
${OPS_SOURCES} [cite: 31, 36, 40]
${HELPERS_SOURCES} ${HELPERS_SOURCES_CUDA} [cite: 31, 34, 38]
${INDEXING_SOURCES} [cite: 32, 35, 40]
${LOOPS_SOURCES} ${LOOPS_SOURCES_CUDA} [cite: 32, 34, 38, 39, 42]
${LEGACY_SOURCES} ${LEGACY_SOURCES_CUDA} [cite: 32, 34, 38]
# Platform sources
${CUSTOMOPS_ONEDNN_SOURCES} [cite: 31, 36, 41]
${CUSTOMOPS_ARMCOMPUTE_SOURCES} [cite: 31, 37, 41] # Include if HAVE_ARMCOMPUTE
${CUSTOMOPS_CUDNN_SOURCES}      # Include if HAVE_CUDNN [cite: 33, 36, 41]
)

# Filter excluded files if not building all ops
if(NOT SD_ALL_OPS)
message("Excluding files based on SD_OPS_LIST for CUDA build...")
# Note: This assumes removeFileIfExcluded works correctly with the ALL_SOURCES list
foreach(SRC_FILE ${ALL_SOURCES})
# Need to call the function carefully - it modifies the list in parent scope
# Maybe better to build a new list?
# removeFileIfExcluded(FILE_ITEM "${SRC_FILE}" LIST_ITEM ALL_SOURCES) # Risky due to loop modification
endforeach()
message(WARNING "Selective op file exclusion for CUDA build needs careful implementation to avoid issues.")
endif()

# Create Object Library (from blas CUDA section)
add_library(samediff_obj OBJECT ${ALL_SOURCES})

# Linker/Compiler settings specific to the object library
target_include_directories(samediff_obj PUBLIC ${ALL_INCLUDE_DIRS})
if(WIN32)
set_property(TARGET samediff_obj PROPERTY MSVC_RUNTIME_LIBRARY "${MSVC_RT_LIB}$<$<CONFIG:Debug>:Debug>") [cite: 43]
endif()

# Create Shared/Static Library
if(NOT SD_STATIC_LIB OR SD_SHARED_LIB)
add_library(${SD_LIBRARY_NAME} SHARED $<TARGET_OBJECTS:samediff_obj>) [cite: 43]
set_property(TARGET ${SD_LIBRARY_NAME} PROPERTY OUTPUT_NAME ${SD_LIBRARY_NAME}) # Ensure correct name
if(WIN32)
set_property(TARGET ${SD_LIBRARY_NAME} PROPERTY MSVC_RUNTIME_LIBRARY "${MSVC_RT_LIB}$<$<CONFIG:Debug>:Debug>") [cite: 45]
endif()
# Link CUDA libraries to the shared library
target_link_libraries(${SD_LIBRARY_NAME} PUBLIC
${CUDA_LIBRARIES}           # cuda runtime etc.
${CUDA_CUBLAS_LIBRARIES}    # cublas
${CUDA_cusolver_LIBRARY}   # cusolver
${CUDA_DRIVER_LIBRARY}      # cuda driver (libcuda) [cite: 45]
${CUDNN}                    # cudnn library (if found) [cite: 45]
${ONEDNN_LIBRARIES}         # oneDNN (if enabled)
${ARMCOMPUTE_LIBRARIES}     # ARM Compute (if enabled)
# Add system libs if needed (pthread etc.)
Threads::Threads # Modern CMake way for pthreads
dl # For dlfcn if needed (e.g., functrace)
${OpenMP_LIBRARIES} # If OpenMP was found
) [cite: 45]
install(TARGETS ${SD_LIBRARY_NAME} DESTINATION lib) # Install to lib dir
endif()

if(SD_STATIC_LIB)
set(STATIC_LIB_NAME ${SD_LIBRARY_NAME})
if(SD_SHARED_LIB) # If building both, append 'static'
set(STATIC_LIB_NAME ${SD_LIBRARY_NAME}static)
endif()
add_library(${STATIC_LIB_NAME} STATIC $<TARGET_OBJECTS:samediff_obj>) [cite: 44]
set_property(TARGET ${STATIC_LIB_NAME} PROPERTY OUTPUT_NAME ${STATIC_LIB_NAME}) # Ensure correct name
if(WIN32)
set_property(TARGET ${STATIC_LIB_NAME} PROPERTY MSVC_RUNTIME_LIBRARY "${MSVC_RT_LIB}$<$<CONFIG:Debug>:Debug>") [cite: 44]
endif()
# Static library doesn't link dependencies here, but they are needed by the final executable linking against it.
install(TARGETS ${STATIC_LIB_NAME} DESTINATION lib) # Install to lib dir [cite: 44]
endif()

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib") # Unified output dir
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib")


# --- CPU Build Path ---
elseif(SD_CPU)
message("--- Configuring CPU Build ---")
add_definitions(-D__CPUBLAS__=true)

# Generate CPU specific compilation units (e.g., helpers, loops)
# Uses genCompilation defined earlier
set(CPU_GENERATED_SOURCES "")
file(GLOB CPU_COMPILATION_UNITS_IN
../include/ops/declarable/helpers/cpu/compilation_units/*.cpp.in
../include/loops/cpu/compilation_units/*.cpp.in
../include/helpers/cpu/loops/*.cpp.in
../include/ops/impl/compilation_units/*.cpp.in # Also include op impl units
)
foreach(FL_ITEM ${CPU_COMPILATION_UNITS_IN})
genCompilation(${FL_ITEM} CPU_GENERATED_SOURCES) # Generate CPU sources
endforeach()

# Generate CPU combination instantiations (using genPartitionCombination)
# Define combinations (from main) [cite: 59, 60]
set(COMBINATIONS_3 "0,0,0;0,0,2;0,1,2;0,2,0;0,2,1;0,2,2;1,0,0;1,1,1;1,1,2;1,2,0;1,2,1;1,2,2;2,0,0;2,0,1;2,0,2;2,1,0;2,1,1;2,1,2;2,2,0;2,2,1;2,2,2") # Semicolon separated list
set(COMBINATIONS_2 "0,0;0,1;1,0;1,1;0,2;2,0;1,2;2,1;2,2") # Semicolon separated list

# Define template files (adjust paths as needed) [cite: 58]
set(INSTANTIATION_TEMPLATES_3 "${CMAKE_CURRENT_SOURCE_DIR}/include/loops/cpu/comb_compilation_units/pairwise_instantiation_template_3.cpp.in") # Adjusted path [cite: 60]
set(INSTANTIATION_TEMPLATES_2 "${CMAKE_CURRENT_SOURCE_DIR}/include/loops/cpu/comb_compilation_units/pairwise_instantiation_template_2.cpp.in") # Adjusted path [cite: 60]

# Define output dir for generated CPU combinations
set(CPU_COMB_GENERATED_DIR "${CMAKE_BINARY_DIR}/cpu_generated_combinations")
file(MAKE_DIRECTORY "${CPU_COMB_GENERATED_DIR}")

# Generate files for combinations_3
if(EXISTS ${INSTANTIATION_TEMPLATES_3})
foreach(COMB ${COMBINATIONS_3})
genPartitionCombination(${INSTANTIATION_TEMPLATES_3} 3 ${COMB} "${CPU_COMB_GENERATED_DIR}")
endforeach()
else()
message(WARNING "CPU Combination Template 3 not found: ${INSTANTIATION_TEMPLATES_3}")
endif()

# Generate files for combinations_2
if(EXISTS ${INSTANTIATION_TEMPLATES_2})
foreach(COMB ${COMBINATIONS_2})
genPartitionCombination(${INSTANTIATION_TEMPLATES_2} 2 ${COMB} "${CPU_COMB_GENERATED_DIR}")
endforeach()
else()
message(WARNING "CPU Combination Template 2 not found: ${INSTANTIATION_TEMPLATES_2}")
endif()
# CUSTOMOPS_GENERIC_SOURCES list is populated by genPartitionCombination

# Gather CPU Source Files (Globs from blas CPU section)
file(GLOB_RECURSE PERF_SOURCES ../include/performance/*.cpp ../include/performance/*.h)
file(GLOB_RECURSE EXCEPTIONS_SOURCES ../include/exceptions/*.cpp ../include/exceptions/*.h)
file(GLOB_RECURSE EXEC_SOURCES ../include/execution/*.cpp ../include/execution/*.h) # Includes impl
file(GLOB_RECURSE TYPES_SOURCES ../include/types/*.cpp ../include/types/*.h)
file(GLOB_RECURSE ARRAY_SOURCES ../include/array/*.cpp ../include/array/*.h) # Includes impl
file(GLOB_RECURSE MEMORY_SOURCES ../include/memory/*.cpp ../include/memory/*.h) # Includes impl
file(GLOB_RECURSE GRAPH_SOURCES ../include/graph/*.cpp ../include/graph/*.h) # Includes impl
file(GLOB_RECURSE CUSTOMOPS_SOURCES ../include/ops/declarable/generic/*.cpp) # Generic ops impl
file(GLOB_RECURSE CUSTOMOPS_HELPERS_CPU ../include/ops/declarable/helpers/cpu/*.cpp) # CPU specific helpers
file(GLOB_RECURSE CUSTOMOPS_HELPERS_IMPL ../include/ops/declarable/helpers/impl/*.cpp) # Common helpers impl
file(GLOB_RECURSE OPS_SOURCES ../include/ops/impl/*.cpp ../include/ops/declarable/impl/*.cpp ../include/ops/*.h)
file(GLOB_RECURSE INDEXING_SOURCES ../include/indexing/*.cpp ../include/indexing/*.h)
file(GLOB_RECURSE HELPERS_SOURCES ../include/build_info.cpp ../include/ConstMessages.cpp ../include/helpers/*.cpp ../include/helpers/*.h) # Common helpers
file(GLOB_RECURSE LEGACY_SOURCES ../include/legacy/impl/*.cpp ../include/legacy/cpu/*.cpp ../include/legacy/*.h) # Legacy impl + CPU specific
file(GLOB_RECURSE LOOPS_SOURCES ../include/loops/*.cpp ../include/loops/*.h) # Includes impl + CPU specific? (Check content)

# Platform specific sources (ONEDNN, ARMCompute)
if(HAVE_ONEDNN)
file(GLOB_RECURSE CUSTOMOPS_ONEDNN_SOURCES ../include/ops/declarable/platform/mkldnn/*.cpp ../include/ops/declarable/platform/mkldnn/mkldnnUtils.h)
endif()
if(HAVE_ARMCOMPUTE) # Check if ARM Compute is actually enabled and found
file(GLOB_RECURSE CUSTOMOPS_ARMCOMPUTE_SOURCES ../include/ops/declarable/platform/armcompute/*.cpp ../include/ops/declarable/platform/armcompute/*.h)
endif()

# Combine all sources for CPU build
set(ALL_SOURCES
${CPU_GENERATED_SOURCES} # Generated from .in files
${CUSTOMOPS_GENERIC_SOURCES} # Generated from combination templates
${PERF_SOURCES}
${EXCEPTIONS_SOURCES}
${EXEC_SOURCES}
${TYPES_SOURCES}
${ARRAY_SOURCES}
${MEMORY_SOURCES}
${GRAPH_SOURCES}
${CUSTOMOPS_SOURCES}
${CUSTOMOPS_HELPERS_CPU}
${CUSTOMOPS_HELPERS_IMPL}
${OPS_SOURCES}
${HELPERS_SOURCES}
${INDEXING_SOURCES}
${LOOPS_SOURCES}
${LEGACY_SOURCES}
# Platform sources
${CUSTOMOPS_ONEDNN_SOURCES}
${CUSTOMOPS_ARMCOMPUTE_SOURCES} # Include if HAVE_ARMCOMPUTE
# Add CALLSTACK_SRC if defined/needed (was in blas CPU add_library)
# ${CALLSTACK_SRC}
)

# Filter excluded files if not building all ops
if(NOT SD_ALL_OPS)
message("Excluding files based on SD_OPS_LIST for CPU build...")
set(FILTERED_SOURCES "")
foreach(SRC_FILE ${ALL_SOURCES})
set(keep_file TRUE)
# Read file and check for NOT_EXCLUDED markers
file(READ ${SRC_FILE} FILE_CONTENTS)
string(FIND "${FILE_CONTENTS}" "NOT_EXCLUDED" NOT_EXCLUDED_IDX)
if(${NOT_EXCLUDED_IDX} GREATER_EQUAL 0)
set(found_matching_op FALSE)
foreach(OP ${SD_OPS_LIST})
string(FIND "${FILE_CONTENTS}" "NOT_EXCLUDED(OP_${OP})" NOT_EXCLUDED_OP_IDX)
if(${NOT_EXCLUDED_OP_IDX} GREATER_EQUAL 0)
set(found_matching_op TRUE)
break()
endif()
endforeach()
if(NOT found_matching_op)
set(keep_file FALSE) # Remove if marker exists but no matching OP found
message(STATUS "Excluding file (no matching NOT_EXCLUDED OP): ${SRC_FILE}")
endif()
endif()
if(keep_file)
list(APPEND FILTERED_SOURCES ${SRC_FILE})
endif()
endforeach()
set(ALL_SOURCES ${FILTERED_SOURCES}) # Replace with filtered list
endif()


# Disable platform optimizations for specific files if needed (from blas CPU section)
if (SD_X86_BUILD AND NOT WIN32) # Only on Linux/macOS x86 builds
# Check if files exist before setting properties
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../include/helpers/impl/OpTracker.cpp")
set_source_files_properties(../include/helpers/impl/OpTracker.cpp PROPERTIES COMPILE_FLAGS "-march=x86-64 -mtune=generic")
endif()
# The NativeOps.cpp file path needs verification - assuming it's under ../include?
# if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../include/cpu/NativeOps.cpp")
#     set_source_files_properties(../include/cpu/NativeOps.cpp PROPERTIES COMPILE_FLAGS "-march=x86-64 -mtune=generic")
# endif()
endif()

# Create Object Library
add_library(samediff_obj OBJECT ${ALL_SOURCES})

# Linker/Compiler settings specific to the object library
target_include_directories(samediff_obj PUBLIC ${ALL_INCLUDE_DIRS})
if(WIN32)
set_property(TARGET samediff_obj PROPERTY MSVC_RUNTIME_LIBRARY "${MSVC_RT_LIB}$<$<CONFIG:Debug>:Debug>")
endif()

# External dependencies (Placeholder from blas - needs definition)
# foreach(external_dependency IN LISTS EXTERNAL_DEPENDENCY_PROJECTS)
#     message("@external ${external_dependency}")
#     add_dependencies(samediff_obj ${external_dependency})
# endforeach()


# Create Shared/Static Library
if(NOT SD_STATIC_LIB OR SD_SHARED_LIB)
add_library(${SD_LIBRARY_NAME} SHARED $<TARGET_OBJECTS:samediff_obj>)
set_property(TARGET ${SD_LIBRARY_NAME} PROPERTY OUTPUT_NAME ${SD_LIBRARY_NAME})
if(WIN32)
set_property(TARGET ${SD_LIBRARY_NAME} PROPERTY MSVC_RUNTIME_LIBRARY "${MSVC_RT_LIB}$<$<CONFIG:Debug>:Debug>")
endif()
if(ANDROID) # Limit jobs for Android builds (from blas)
cmake_host_system_information(RESULT _logical_cores QUERY NUMBER_OF_LOGICAL_CORES)
if(_logical_cores LESS 4)
set_target_properties(${SD_LIBRARY_NAME} PROPERTIES JOB_POOL_COMPILE one_jobs)
endif()
endif()
# Link dependencies to the shared library
target_link_libraries(${SD_LIBRARY_NAME} PUBLIC
${BLAS_LIBRARIES}           # System BLAS found earlier
${OPENBLAS_LIBRARIES}       # External OpenBLAS if used
${ONEDNN_LIBRARIES}         # oneDNN (if enabled)
${ARMCOMPUTE_LIBRARIES}     # ARM Compute (if enabled)
# ${CPU_FEATURES}          # cpu_features lib (if used/needed, requires find_package)
Threads::Threads            # Pthreads
dl                          # dlfcn
${OpenMP_LIBRARIES}         # OpenMP libs (if found)
# Add other external dependency libs here
# ${EXTERNAL_DEPENDENCY_LIBS}
)
install(TARGETS ${SD_LIBRARY_NAME} DESTINATION lib)
endif()

if(SD_STATIC_LIB)
set(STATIC_LIB_NAME ${SD_LIBRARY_NAME})
if(SD_SHARED_LIB)
set(STATIC_LIB_NAME ${SD_LIBRARY_NAME}static)
endif()
add_library(${STATIC_LIB_NAME} STATIC $<TARGET_OBJECTS:samediff_obj>)
set_property(TARGET ${STATIC_LIB_NAME} PROPERTY OUTPUT_NAME ${STATIC_LIB_NAME})
if(WIN32)
set_property(TARGET ${STATIC_LIB_NAME} PROPERTY MSVC_RUNTIME_LIBRARY "${MSVC_RT_LIB}$<$<CONFIG:Debug>:Debug>")
endif()
# Static lib doesn't link dependencies here
install(TARGETS ${STATIC_LIB_NAME} DESTINATION lib)
endif()

# Compiler Checks (from blas)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.9)
message(FATAL_ERROR "You need at least GCC 4.9 for CPU build.")
endif()

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib") # Unified output dir
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib")

else()
message(FATAL_ERROR "Neither SD_CUDA nor SD_CPU is set. Cannot determine build target.")
endif() # End of SD_CUDA / SD_CPU blocks


# --- Preprocessing Step (Optional - from blas) ---
if(SD_PREPROCESS STREQUAL "ON")
message("--- Running Preprocessing Step ---")
message("Preprocessing enabled: Output will be in ${CMAKE_CURRENT_SOURCE_DIR}/preprocessed") # Adjusted path message

# Ensure ALL_SOURCES is populated from the correct build path (CUDA or CPU)
if(NOT ALL_SOURCES)
message(FATAL_ERROR "SD_PREPROCESS=ON but ALL_SOURCES list is empty. Ensure CUDA or CPU build ran first.")
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include) # Ensure base include is present
list(REMOVE_DUPLICATES ALL_SOURCES) # Remove duplicates before processing

set(PREPROCESSED_DIR "${CMAKE_CURRENT_SOURCE_DIR}/preprocessed") # Output dir relative to source
file(MAKE_DIRECTORY ${PREPROCESSED_DIR})

set(PREPROCESSED_FILES "")
set(PROCESSED_SOURCES_FOR_PREPROC "") # Use a separate list to track processed files in this step

# Gather necessary include directories for preprocessing
set(PREPROC_INCLUDE_DIRS "")
get_property(target_includes TARGET samediff_obj PROPERTY INCLUDE_DIRECTORIES) # Get includes from the object lib
if(target_includes)
foreach(dir ${target_includes})
# Check if directory exists, CMake properties can sometimes list non-existent paths
if(IS_DIRECTORY "${dir}")
list(APPEND PREPROC_INCLUDE_DIRS "-I${dir}")
endif()
endforeach()
else()
# Fallback to manual list if property fails (adjust paths as needed)
list(APPEND PREPROC_INCLUDE_DIRS
"-I${CMAKE_CURRENT_SOURCE_DIR}/include"
"-I${CMAKE_CURRENT_BINARY_DIR}/include"
# Add other essential include paths manually if needed
# "-I${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-src/include" # Example
)
endif()
string(REPLACE ";" " " include_dirs_str "${PREPROC_INCLUDE_DIRS}")


# Get compiler executable
if(SD_CUDA AND CMAKE_CUDA_COMPILER)
set(compiler "${CMAKE_CUDA_COMPILER}")
set(compile_flags "${CMAKE_CUDA_FLAGS}") # Use CUDA flags
elseif(CMAKE_CXX_COMPILER)
set(compiler "${CMAKE_CXX_COMPILER}")
set(compile_flags "${CMAKE_CXX_FLAGS}") # Use CXX flags
else()
message(FATAL_ERROR "Cannot determine compiler for preprocessing.")
endif()

# Get compile definitions
get_property(target_defs TARGET samediff_obj PROPERTY COMPILE_DEFINITIONS)
set(defs_str "")
if(target_defs)
foreach(d ${target_defs})
list(APPEND defs_str "-D${d}")
endforeach()
endif()
string(REPLACE ";" " " defs_str "${defs_str}")

message("Preprocessing Include Dirs: ${include_dirs_str}")
message("Preprocessing Definitions: ${defs_str}")

foreach(src IN LISTS ALL_SOURCES)
# Only preprocess actual source files, not headers included in the list
if(src MATCHES "\\.(c|cpp|cxx|cc|cu)$")
if(NOT src IN_LIST PROCESSED_SOURCES_FOR_PREPROC)
get_filename_component(src_name ${src} NAME)
get_filename_component(src_path ${src} ABSOLUTE) # Get absolute path

# Create a unique name for the preprocessed file based on relative path
string(REPLACE "${CMAKE_SOURCE_DIR}/" "" rel_path ${src_path}) # Relative path
string(REPLACE "/" "_" flat_rel_path ${rel_path}) # Flatten path
set(preprocessed_file "${PREPROCESSED_DIR}/${flat_rel_path}.i")

message(STATUS "Preprocessing ${src} to ${preprocessed_file}")

if(NOT EXISTS "${preprocessed_file}")
# Construct command
set(preprocess_command "${compiler}" "-E" ${compile_flags} ${include_dirs_str} ${defs_str} "${src_path}" "-o" "${preprocessed_file}")
# message("COMMAND: ${preprocess_command}") # Debug output

execute_process(
COMMAND ${preprocess_command}
# WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/../" # Working dir from blas, may need adjustment
WORKING_DIRECTORY "${CMAKE_BINARY_DIR}" # Safer working directory
RESULT_VARIABLE result
OUTPUT_VARIABLE stdout_output
ERROR_VARIABLE stderr_output
OUTPUT_STRIP_TRAILING_WHITESPACE
ERROR_STRIP_TRAILING_WHITESPACE
)

if(NOT result EQUAL 0)
message(WARNING "Preprocessing failed for ${src} (Result: ${result}). Error:\n${stderr_output}")
# Optionally log output even on failure
# set(output_log_file "${preprocessed_file}.error")
# file(WRITE ${output_log_file} "COMMAND: ${preprocess_command}\n\nStandard Output:\n${stdout_output}\n\nStandard Error:\n${stderr_output}\n")
else()
# Optionally save command/output on success
# set(output_log_file "${preprocessed_file}.log")
# file(WRITE ${output_log_file} "COMMAND: ${preprocess_command}\n\nStandard Output:\n${stdout_output}\n\nStandard Error:\n${stderr_output}\n")
list(APPEND PREPROCESSED_FILES ${preprocessed_file})
endif()
else()
message(STATUS "Preprocessed file already exists, skipping: ${preprocessed_file}")
list(APPEND PREPROCESSED_FILES ${preprocessed_file}) # Add existing file to list
endif()

list(APPEND PROCESSED_SOURCES_FOR_PREPROC ${src})
else()
# message(STATUS "Skipping already processed file: ${src}")
endif()
endif()
endforeach()

set_directory_properties(PROPERTIES CLEAN_NO_CUSTOM 1)

# Create a Custom Target for All Preprocessed Files
if(PREPROCESSED_FILES)
add_custom_target(preprocess_sources ALL DEPENDS ${PREPROCESSED_FILES} COMMENT "Preprocessing source files")
endif()
endif() # End SD_PREPROCESS


# --- Testing Setup (Optional - from main) ---
if(SD_BUILD_TESTS)
message("--- Configuring Tests ---")
include(CTest)
enable_testing()

# Assume tests are in a 'tests' subdirectory relative to this CMakeLists.txt
if(IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/tests")
# Add test subdirectory - it should define its own tests and link against ${SD_LIBRARY_NAME}
# The tests subdirectory CMakeLists.txt might need adjustments if it relied on variables from the old main/blas structure
add_subdirectory(tests)
else()
message(WARNING "SD_BUILD_TESTS is ON, but the 'tests' subdirectory was not found.")
endif()
# Example of adding a test directly (if tests/CMakeLists.txt doesn't exist)
# add_executable(my_test tests/my_test.cpp)
# target_link_libraries(my_test PRIVATE ${SD_LIBRARY_NAME} GTest::gtest GTest::gtest_main) # Link against built library and GTest
# add_test(NAME MyTest COMMAND my_test)
endif()

# --- Installation and Packaging (from main) ---
message("--- Configuring Installation and Packaging ---")

# Set install prefix (adjust as needed)
if(NOT CMAKE_INSTALL_PREFIX)
set(CMAKE_INSTALL_PREFIX "/usr/local" CACHE PATH "Installation prefix")
endif()
message("Installation prefix: ${CMAKE_INSTALL_PREFIX}")

# Install libraries (already added install commands in CUDA/CPU blocks)
# Install header files (example)
install(DIRECTORY include/ DESTINATION include) # Install public headers

# CPack Configuration (from main) [cite: 117, 118, 119]
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Native operations for nd4j.")
set(CPACK_PACKAGE_RELEASE 1) # Or derive from version string
set(CPACK_PACKAGE_CONTACT "agibsonccc <adam@skymind.global>")
set(CPACK_PACKAGE_VENDOR "Eclipse DL4J Project") # Updated vendor
set(CPACK_PACKAGE_NAME "libnd4j")
# Try to get version from project version or define manually
if(PROJECT_VERSION)
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
else()
set(CPACK_PACKAGE_VERSION "1.0.0") # Fallback version
endif()
set(CPACK_PACKAGE_VERSION_MAJOR ${CPACK_PACKAGE_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${CPACK_PACKAGE_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${CPACK_PACKAGE_VERSION_PATCH})
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE") # Point to license file
set(CPACK_RESOURCE_FILE_README "${CMAKE_CURRENT_SOURCE_DIR}/README.md")
set(CPACK_PACKAGE_INSTALL_DIRECTORY "${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}") # Installation dir within prefix

# Generator specific settings
if(UNIX)
# Detect Linux distribution
find_program(LSB_RELEASE_EXEC lsb_release)
if(LSB_RELEASE_EXEC)
execute_process(COMMAND ${LSB_RELEASE_EXEC} -si OUTPUT_VARIABLE DISTRIBUTION OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND ${LSB_RELEASE_EXEC} -sc OUTPUT_VARIABLE RELEASE OUTPUT_STRIP_TRAILING_WHITESPACE)
else()
# Fallback detection (e.g., check /etc/os-release)
if(EXISTS "/etc/os-release")
file(STRINGS "/etc/os-release" OS_ID REGEX "^ID=.*")
string(REPLACE "ID=" "" DISTRIBUTION "${OS_ID}")
# Release detection might be harder here
endif()
endif()

# Detect architecture
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
set(ARCHITECTURE "amd64") # Debian naming
set(RPM_ARCHITECTURE "x86_64") # RPM naming
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
set(ARCHITECTURE "arm64")
set(RPM_ARCHITECTURE "aarch64")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "armv7")
set(ARCHITECTURE "armhf") # Example, might depend on distro
set(RPM_ARCHITECTURE "armv7hl") # Example
else()
set(ARCHITECTURE ${CMAKE_SYSTEM_PROCESSOR})
set(RPM_ARCHITECTURE ${CMAKE_SYSTEM_PROCESSOR})
endif()

if(DISTRIBUTION MATCHES "Ubuntu|Debian")
set(CPACK_GENERATOR "DEB")
set(CPACK_PACKAGE_FILE_NAME ${CPACK_PACKAGE_NAME}_${CPACK_PACKAGE_VERSION}-${CPACK_PACKAGE_RELEASE}_${ARCHITECTURE}) [cite: 117]
set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE ${ARCHITECTURE}) [cite: 117]
set(CPACK_DEBIAN_PACKAGE_MAINTAINER ${CPACK_PACKAGE_CONTACT}) [cite: 117]
set(CPACK_DEBIAN_PACKAGE_SECTION "libs") # Changed from devel [cite: 117]
set(CPACK_DEBIAN_PACKAGE_HOMEPAGE "https://github.com/eclipse/deeplearning4j") # Updated URL [cite: 117]
# Define dependencies carefully based on actual needs
# set(CPACK_DEBIAN_PACKAGE_DEPENDS "libc6 (>= 2.17), libstdc++6 (>= 5)") # Example basic deps
# Add CUDA/cuDNN deps if built with them
if(SD_CUDA) list(APPEND CPACK_DEBIAN_PACKAGE_DEPENDS "cuda-cudart-${CUDA_VERSION_MAJOR}-${CUDA_VERSION_MINOR}") # Example
if(HAVE_CUDNN) list(APPEND CPACK_DEBIAN_PACKAGE_DEPENDS "libcudnn8") # Example, version might vary
string(REPLACE ";" ", " CPACK_DEBIAN_PACKAGE_DEPENDS "${CPACK_DEBIAN_PACKAGE_DEPENDS}")

# Postinst/Postrm scripts (Ensure these exist and are correct) [cite: 118]
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cmake/postinst.deb")
set(CPACK_DEBIAN_PACKAGE_CONTROL_EXTRA "${CMAKE_CURRENT_SOURCE_DIR}/cmake/postinst.deb;${CMAKE_CURRENT_SOURCE_DIR}/cmake/postrm.deb")
endif()
elseif(DISTRIBUTION MATCHES "CentOS|Fedora|RHEL")
set(CPACK_GENERATOR "RPM")
set(CPACK_PACKAGE_FILE_NAME ${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}-${CPACK_PACKAGE_RELEASE}.${RPM_ARCHITECTURE}) [cite: 118]
set(CPACK_RPM_PACKAGE_ARCHITECTURE ${RPM_ARCHITECTURE}) [cite: 118]
set(CPACK_RPM_PACKAGE_LICENSE "Apache-2.0") [cite: 118]
set(CPACK_RPM_PACKAGE_GROUP "System Environment/Libraries") # Changed from Development/Tools [cite: 118]
set(CPACK_RPM_PACKAGE_URL "https://github.com/eclipse/deeplearning4j") # Updated URL [cite: 118]
# Define dependencies carefully
# set(CPACK_RPM_PACKAGE_REQUIRES "glibc >= 2.17, libstdc++ >= 4.8") # Example basic deps
# Add CUDA/cuDNN deps if built with them
if(SD_CUDA) list(APPEND CPACK_RPM_PACKAGE_REQUIRES "cuda-cudart-${CUDA_VERSION_MAJOR}-${CUDA_VERSION_MINOR}") # Example
if(HAVE_CUDNN) list(APPEND CPACK_RPM_PACKAGE_REQUIRES "libcudnn8") # Example, version might vary
string(REPLACE ";" ", " CPACK_RPM_PACKAGE_REQUIRES "${CPACK_RPM_PACKAGE_REQUIRES}")

# Postinst/Postrm scripts [cite: 119]
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cmake/postinst.rpm")
set(CPACK_RPM_POST_INSTALL_SCRIPT_FILE "${CMAKE_CURRENT_SOURCE_DIR}/cmake/postinst.rpm")
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cmake/postrm.rpm")
set(CPACK_RPM_POST_UNINSTALL_SCRIPT_FILE "${CMAKE_CURRENT_SOURCE_DIR}/cmake/postrm.rpm")
endif()
# Exclude auto filelist addition for common paths if needed [cite: 119]
# set(CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION "/usr/local/lib")
else()
# Default generator for other Unix-like systems (e.g., TGZ)
set(CPACK_GENERATOR "TGZ")
endif()
elseif(WIN32)
set(CPACK_GENERATOR "NSIS") # Or ZIP
set(CPACK_PACKAGE_INSTALL_DIRECTORY "${CPACK_PACKAGE_NAME} ${CPACK_PACKAGE_VERSION}")
set(CPACK_NSIS_DISPLAY_NAME "${CPACK_PACKAGE_NAME}")
set(CPACK_NSIS_PUBLISHER "${CPACK_PACKAGE_VENDOR}")
set(CPACK_NSIS_PACKAGE_NAME "${CPACK_PACKAGE_NAME} ${CPACK_PACKAGE_VERSION}")
elseif(APPLE)
set(CPACK_GENERATOR "DragNDrop") # Or PKG, TGZ
set(CPACK_PACKAGE_FILE_NAME ${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}-macOS)
endif()

include(CPack)

message("--- CMake Configuration Complete ---")