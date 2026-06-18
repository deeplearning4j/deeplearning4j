# android-arm64.cmake - Intel x86_64 to ARM64 cross-compilation toolchain

# Set the system and processor for cross-compilation
set(CMAKE_SYSTEM_NAME Android)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

# Flexible API level - can be overridden via command line or environment
if(NOT DEFINED ANDROID_NATIVE_API_LEVEL AND DEFINED ENV{ANDROID_VERSION})
   set(ANDROID_NATIVE_API_LEVEL $ENV{ANDROID_VERSION})
elseif(NOT DEFINED ANDROID_NATIVE_API_LEVEL)
   set(ANDROID_NATIVE_API_LEVEL 21)  # Default API level
endif()

set(CMAKE_SYSTEM_VERSION ${ANDROID_NATIVE_API_LEVEL})
set(CMAKE_ANDROID_ARCH_ABI arm64-v8a)

# Flexible NDK path detection
if(NOT DEFINED CMAKE_ANDROID_NDK)
   if(DEFINED ENV{ANDROID_NDK_ROOT})
      set(CMAKE_ANDROID_NDK $ENV{ANDROID_NDK_ROOT})
   elseif(DEFINED ENV{ANDROID_NDK_HOME})
      set(CMAKE_ANDROID_NDK $ENV{ANDROID_NDK_HOME})
   elseif(DEFINED ENV{ANDROID_NDK})
      set(CMAKE_ANDROID_NDK $ENV{ANDROID_NDK})
   else()
      message(FATAL_ERROR "Android NDK not found. Please set ANDROID_NDK_ROOT, ANDROID_NDK_HOME, or ANDROID_NDK environment variable")
   endif()
endif()

# Verify NDK exists
if(NOT EXISTS "${CMAKE_ANDROID_NDK}")
   message(FATAL_ERROR "Android NDK directory does not exist: ${CMAKE_ANDROID_NDK}")
endif()

# Set Android STL
set(CMAKE_ANDROID_STL_TYPE c++_shared)

# Set NDK toolchain paths for Intel x86_64 host
set(NDK_TOOLCHAIN_PATH "${CMAKE_ANDROID_NDK}/toolchains/llvm/prebuilt/linux-x86_64")

# Check if the toolchain path exists
if(NOT EXISTS "${NDK_TOOLCHAIN_PATH}")
   message(FATAL_ERROR "NDK toolchain path does not exist: ${NDK_TOOLCHAIN_PATH}")
endif()

# Set sysroot
set(CMAKE_SYSROOT "${NDK_TOOLCHAIN_PATH}/sysroot")

# Verify sysroot exists
if(NOT EXISTS "${CMAKE_SYSROOT}")
   message(FATAL_ERROR "NDK sysroot does not exist: ${CMAKE_SYSROOT}")
endif()

# Use NDK compilers for cross-compilation
set(CMAKE_C_COMPILER "${NDK_TOOLCHAIN_PATH}/bin/aarch64-linux-android${ANDROID_NATIVE_API_LEVEL}-clang")
set(CMAKE_CXX_COMPILER "${NDK_TOOLCHAIN_PATH}/bin/aarch64-linux-android${ANDROID_NATIVE_API_LEVEL}-clang++")
set(CMAKE_AR "${NDK_TOOLCHAIN_PATH}/bin/llvm-ar")
set(CMAKE_STRIP "${NDK_TOOLCHAIN_PATH}/bin/llvm-strip")
set(CMAKE_RANLIB "${NDK_TOOLCHAIN_PATH}/bin/llvm-ranlib")

# Verify compilers exist
if(NOT EXISTS "${CMAKE_C_COMPILER}")
   message(FATAL_ERROR "C compiler does not exist: ${CMAKE_C_COMPILER}")
endif()

if(NOT EXISTS "${CMAKE_CXX_COMPILER}")
   message(FATAL_ERROR "C++ compiler does not exist: ${CMAKE_CXX_COMPILER}")
endif()

# Set cross-compilation flags
# Use armv8.2-a for dotprod support (required for MLIR ARM optimized kernels)
# C++17 required for MLIR/LLVM headers
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC -march=armv8.2-a+dotprod")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -march=armv8.2-a+dotprod -std=c++17")

# Set NDK library paths for linking
set(NDK_SYSROOT_LIB_PATH "${CMAKE_SYSROOT}/usr/lib/aarch64-linux-android/${ANDROID_NATIVE_API_LEVEL}")
set(NDK_TOOLCHAIN_LIB_PATH "${NDK_TOOLCHAIN_PATH}/lib64/clang/14.0.6/lib/linux")

# Linker flags
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--build-id=sha1")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-rosegment")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--fatal-warnings")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--gc-sections")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-undefined")

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--build-id=sha1")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--no-rosegment")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--fatal-warnings")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--gc-sections")

# Additional Android-specific flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -ffunction-sections -fdata-sections")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffunction-sections -fdata-sections")

# Set the find root path for cross-compilation
set(CMAKE_FIND_ROOT_PATH "${CMAKE_SYSROOT}" "${CMAKE_ANDROID_NDK}")
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# Configure for cross-compilation
set(CMAKE_CROSSCOMPILING TRUE)
set(CMAKE_CROSSCOMPILING_EMULATOR "")

# Android specific settings
set(ANDROID TRUE)
set(ANDROID_ABI "arm64-v8a")
set(ANDROID_PLATFORM "android-${ANDROID_NATIVE_API_LEVEL}")
set(ANDROID_STL "c++_shared")

# Debug information
message(STATUS "Cross-compiling from Intel x86_64 to ARM64")
message(STATUS "Android NDK: ${CMAKE_ANDROID_NDK}")
message(STATUS "Android API Level: ${ANDROID_NATIVE_API_LEVEL}")
message(STATUS "Android ABI: ${CMAKE_ANDROID_ARCH_ABI}")
message(STATUS "Android Sysroot: ${CMAKE_SYSROOT}")
message(STATUS "C Compiler: ${CMAKE_C_COMPILER}")
message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "Toolchain Path: ${NDK_TOOLCHAIN_PATH}")

# Ensure we don't accidentally use host tools
set(CMAKE_C_COMPILER_WORKS 1)
set(CMAKE_CXX_COMPILER_WORKS 1)
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

# Set compiler identification to prevent issues
set(CMAKE_C_COMPILER_ID "Clang")
set(CMAKE_CXX_COMPILER_ID "Clang")

# Additional search paths for Android-specific libraries
if(EXISTS "${NDK_SYSROOT_LIB_PATH}")
   set(CMAKE_LIBRARY_PATH "${CMAKE_LIBRARY_PATH}:${NDK_SYSROOT_LIB_PATH}")
endif()

if(EXISTS "${NDK_TOOLCHAIN_LIB_PATH}")
   set(CMAKE_LIBRARY_PATH "${CMAKE_LIBRARY_PATH}:${NDK_TOOLCHAIN_LIB_PATH}")
endif()

# Set Android-specific preprocessor definitions
add_definitions(-DANDROID -D__ANDROID__ -D__ANDROID_API__=${ANDROID_NATIVE_API_LEVEL})

# Optimization flags for ARM64
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")


# --- MLIR Cross-Compilation Support ---
# When HELPERS_mlir=ON for Android builds, MLIR AOT compilation is used:
# - Kernels are compiled on the host x86_64 machine targeting aarch64-linux-android
# - Pre-compiled .o files are linked into the final .so (no LLVM JIT on device)
# - The MLIR AArch64 backend is auto-enabled for cross-compilation
#
# To build with MLIR support for Android:
#   cmake -DCMAKE_TOOLCHAIN_FILE=cmake/android-arm64.cmake \
#         -DHELPERS_mlir=ON \
#         -DMLIR_ENABLE_VULKAN=ON \  # Optional: enable Vulkan GPU offload
#         -DLLVM_DIR=/path/to/host/llvm \
#         ..
#
# Note: LLVM_DIR should point to the HOST LLVM installation (not cross-compiled).
# The host LLVM/MLIR is used for AOT compilation, producing ARM64 object code.
if(HELPERS_mlir)
    message(STATUS "MLIR: Cross-compilation mode for Android ARM64")
    message(STATUS "  Kernels will be AOT-compiled on host targeting aarch64-linux-android")
    message(STATUS "  ARM features: NEON, dotprod (ARMv8.2-A)")
endif()

# --- NNAPI Support ---
# Auto-enable NNAPI for Android API 27+ (NNAPI 1.0)
# NNAPI routes DSP segments to hardware accelerators (Hexagon DSP, GPU, NPU)
if(ANDROID_NATIVE_API_LEVEL GREATER_EQUAL 27)
    set(HELPERS_nnapi ON CACHE BOOL "Auto-enabled for Android API 27+" FORCE)
    set(HAVE_NNAPI ON CACHE BOOL "NNAPI available" FORCE)
    add_definitions(-DHAVE_NNAPI=1)

    # Link against NNAPI shared library (libneuralnetworks.so)
    # Available in the NDK sysroot from API 27+
    set(NNAPI_LINK_FLAGS "-lneuralnetworks")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${NNAPI_LINK_FLAGS}")

    message(STATUS "NNAPI: Enabled for Android API ${ANDROID_NATIVE_API_LEVEL}")
    if(ANDROID_NATIVE_API_LEVEL GREATER_EQUAL 30)
        message(STATUS "  NNAPI 1.3: batch_matmul, quantized ops, control flow")
    elseif(ANDROID_NATIVE_API_LEVEL GREATER_EQUAL 29)
        message(STATUS "  NNAPI 1.2: comparison ops, logical ops, reduce ops")
    else()
        message(STATUS "  NNAPI 1.0: basic elementwise, conv, pooling, softmax")
    endif()
else()
    message(STATUS "NNAPI: Disabled (API ${ANDROID_NATIVE_API_LEVEL} < 27)")
endif()

message(STATUS "Android ARM64 cross-compilation toolchain configured successfully")