
# android-arm64.cmake - A robust and simplified CMake toolchain file for
# Android NDK r21+ targeting arm64-v8a.
#
# This file provides a single source of truth for compiler and toolchain paths,
# preventing conflicts and ensuring a reliable cross-compilation environment.
#
################################################################################

# --- Phase 1: Core Environment Setup ---
# Set the target system and architecture. This is non-negotiable for this toolchain.
set(CMAKE_SYSTEM_NAME Android)
set(CMAKE_ANDROID_ARCH_ABI "arm64-v8a")

# --- Phase 2: NDK and API Level Configuration ---
# Set the Android NDK path, failing if it's not explicitly defined.
if(DEFINED ENV{ANDROID_NDK_ROOT})
   set(CMAKE_ANDROID_NDK "$ENV{ANDROID_NDK_ROOT}")
elseif(DEFINED ENV{ANDROID_NDK})
   set(CMAKE_ANDROID_NDK "$ENV{ANDROID_NDK}")
else()
   message(FATAL_ERROR "❌ Android NDK path not found. Set the ANDROID_NDK or ANDROID_NDK_ROOT environment variable.")
endif()

# Verify the NDK directory exists.
if(NOT EXISTS "${CMAKE_ANDROID_NDK}")
   message(FATAL_ERROR "❌ Android NDK directory does not exist: ${CMAKE_ANDROID_NDK}")
endif()

# Set the API level. Default to 21 if not specified.
if(DEFINED ENV{ANDROID_VERSION})
   set(ANDROID_NATIVE_API_LEVEL "$ENV{ANDROID_VERSION}")
else()
   set(ANDROID_NATIVE_API_LEVEL 21)
endif()
set(CMAKE_SYSTEM_VERSION ${ANDROID_NATIVE_API_LEVEL})
set(CMAKE_ANDROID_NDK_TOOLCHAIN_VERSION "clang")
set(CMAKE_ANDROID_STL_TYPE "c++_shared")

# --- Phase 3: Host and Toolchain Path Detection (Single Source of Truth) ---
# Detect the host OS and architecture to determine the correct NDK toolchain path.
# This is the single point where the host-specific path is determined.
if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux")
   if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
      set(NDK_HOST_TAG "linux-aarch64")
   else()
      set(NDK_HOST_TAG "linux-x86_64")
   endif()
elseif(CMAKE_HOST_SYSTEM_NAME STREQUAL "Darwin")
   set(NDK_HOST_TAG "darwin-x86_64")
elseif(CMAKE_HOST_SYSTEM_NAME STREQUAL "Windows")
   set(NDK_HOST_TAG "windows-x86_64")
else()
   # Default fallback for unknown systems.
   set(NDK_HOST_TAG "linux-x86_64")
endif()

# Construct the definitive toolchain and sysroot paths.
set(NDK_TOOLCHAIN_PATH "${CMAKE_ANDROID_NDK}/toolchains/llvm/prebuilt/${NDK_HOST_TAG}")
set(CMAKE_SYSROOT "${NDK_TOOLCHAIN_PATH}/sysroot")

# Verify that the determined toolchain paths are valid.
if(NOT EXISTS "${NDK_TOOLCHAIN_PATH}")
   message(FATAL_ERROR "❌ NDK toolchain path does not exist: ${NDK_TOOLCHAIN_PATH}")
endif()
if(NOT EXISTS "${CMAKE_SYSROOT}")
   message(FATAL_ERROR "❌ NDK sysroot does not exist: ${CMAKE_SYSROOT}")
endif()


# --- Phase 4: Compiler and Linker Configuration ---
# Set the C and CXX compilers using the definitive toolchain path.
# This is the ONLY place these variables are set.
set(CMAKE_C_COMPILER "${NDK_TOOLCHAIN_PATH}/bin/aarch64-linux-android${ANDROID_NATIVE_API_LEVEL}-clang")
set(CMAKE_CXX_COMPILER "${NDK_TOOLCHAIN_PATH}/bin/aarch64-linux-android${ANDROID_NATIVE_API_LEVEL}-clang++")

# Verify the compilers exist at the specified paths.
if(NOT EXISTS "${CMAKE_C_COMPILER}")
   message(FATAL_ERROR "❌ C compiler not found at: ${CMAKE_C_COMPILER}")
endif()
if(NOT EXISTS "${CMAKE_CXX_COMPILER}")
   message(FATAL_ERROR "❌ C++ compiler not found at: ${CMAKE_CXX_COMPILER}")
endif()

# Set the find root path to the NDK sysroot to ensure correct library/header discovery.
set(CMAKE_FIND_ROOT_PATH "${CMAKE_SYSROOT}")
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)


# --- Phase 5: Compile and Link Flags ---
# Set essential compile and link flags for the Android armv8-a target.
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=armv8-a -fPIC -ffunction-sections -fdata-sections -fstack-protector-strong")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8-a -fPIC -ffunction-sections -fdata-sections -fstack-protector-strong")

# Add definitions required for the build.
add_definitions(-D__ANDROID_API__=${ANDROID_NATIVE_API_LEVEL} -DANDROID)

# Configure linker flags for garbage collection to reduce binary size.
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--gc-sections")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--gc-sections")


# --- Phase 6: Final Verification and Output ---
# Print a summary of the configured build environment for easy debugging.
message(STATUS "✅ Android Toolchain Configured Successfully")
message(STATUS "   NDK Path: ${CMAKE_ANDROID_NDK}")
message(STATUS "   Host Tag: ${NDK_HOST_TAG}")
message(STATUS "   API Level: ${ANDROID_NATIVE_API_LEVEL}")
message(STATUS "   ABI: ${CMAKE_ANDROID_ARCH_ABI}")
message(STATUS "   Sysroot: ${CMAKE_SYSROOT}")
message(STATUS "   C Compiler: ${CMAKE_C_COMPILER}")
message(STATUS "   C++ Compiler: ${CMAKE_CXX_COMPILER}")
