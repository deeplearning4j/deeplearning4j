# android-arm64.cmake - CMake toolchain for Android ARM64 cross-compilation
# Designed to work with non-standard ARM64-hosted NDK (Termux NDK)

# Set target system
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

# Get NDK root from environment
set(ANDROID_NDK_ROOT $ENV{ANDROID_NDK_ROOT})
if(NOT ANDROID_NDK_ROOT)
   message(FATAL_ERROR "ANDROID_NDK_ROOT environment variable is not set.")
endif()

# Set paths
set(TOOLCHAIN_DIR "${ANDROID_NDK_ROOT}/toolchains/llvm/prebuilt/linux-aarch64")
set(CMAKE_SYSROOT "${TOOLCHAIN_DIR}/sysroot")

# Debug: Print environment variables
message(STATUS "ANDROID_NDK_ROOT: ${ANDROID_NDK_ROOT}")
message(STATUS "TOOLCHAIN_DIR: ${TOOLCHAIN_DIR}")
message(STATUS "Environment CMAKE_C_COMPILER: $ENV{CMAKE_C_COMPILER}")
message(STATUS "Environment CMAKE_CXX_COMPILER: $ENV{CMAKE_CXX_COMPILER}")

# Set C compiler - prioritize environment variables from workflow
if(DEFINED ENV{CMAKE_C_COMPILER} AND NOT "$ENV{CMAKE_C_COMPILER}" STREQUAL "")
   set(CMAKE_C_COMPILER $ENV{CMAKE_C_COMPILER})
   message(STATUS "Using C compiler from environment: ${CMAKE_C_COMPILER}")
else()
   # Fall back to direct clang binary
   set(CMAKE_C_COMPILER "${TOOLCHAIN_DIR}/bin/clang")
   message(STATUS "Using direct clang binary: ${CMAKE_C_COMPILER}")
endif()

# Set C++ compiler - ALWAYS use clang for C++ since clang++ doesn't exist in Termux NDK
set(CMAKE_CXX_COMPILER "${CMAKE_C_COMPILER}")
message(STATUS "Using C compiler for C++ (clang++ not available in Termux NDK): ${CMAKE_CXX_COMPILER}")

# Set ASM compiler
set(CMAKE_ASM_COMPILER ${CMAKE_C_COMPILER})

# Set cross-compilation behavior
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH ${CMAKE_SYSROOT})

# Set Android-specific compiler flags
set(CMAKE_C_FLAGS "--target=aarch64-linux-android21 --sysroot=${CMAKE_SYSROOT}" CACHE STRING "C compiler flags")
set(CMAKE_CXX_FLAGS "--target=aarch64-linux-android21 --sysroot=${CMAKE_SYSROOT} -stdlib=libc++" CACHE STRING "C++ compiler flags")
set(CMAKE_ASM_FLAGS "--target=aarch64-linux-android21 --sysroot=${CMAKE_SYSROOT}" CACHE STRING "Assembler flags")

# Debug: Print final compiler selections
message(STATUS "Final C compiler: ${CMAKE_C_COMPILER}")
message(STATUS "Final C++ compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "Final ASM compiler: ${CMAKE_ASM_COMPILER}")
message(STATUS "C flags: ${CMAKE_C_FLAGS}")
message(STATUS "CXX flags: ${CMAKE_CXX_FLAGS}")