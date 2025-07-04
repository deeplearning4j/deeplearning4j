# android-arm64.cmake - Manual Toolchain for non-standard ARM64-hosted NDK

# This method avoids CMake's automatic discovery, which fails on the Termux NDK,
# by explicitly defining the compilers and system root.

# 1. Set the target system processor. Setting CMAKE_SYSTEM_NAME to a generic
#    value like Linux prevents CMake from trying to run its broken Android module.
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

# 2. Point to the NDK root from the environment variable set in your workflow.
set(ANDROID_NDK_ROOT $ENV{ANDROID_NDK_ROOT})
if(NOT ANDROID_NDK_ROOT)
   message(FATAL_ERROR "ANDROID_NDK_ROOT environment variable is not set.")
endif()

# 3. Manually define the paths to the toolchain directory and the system root.
#    This is the most critical part of the fix.
set(TOOLCHAIN_DIR "${ANDROID_NDK_ROOT}/toolchains/llvm/prebuilt/linux-aarch64")
set(CMAKE_SYSROOT "${TOOLCHAIN_DIR}/sysroot")

# 4. Explicitly set the C and C++ compilers for your target API level (21).
#    This resolves the "CMAKE_CXX_COMPILER not set" error.
set(CMAKE_C_COMPILER "${TOOLCHAIN_DIR}/bin/aarch64-linux-android21-clang")
set(CMAKE_CXX_COMPILER "${TOOLCHAIN_DIR}/bin/aarch64-linux-android21-clang++")
set(CMAKE_ASM_COMPILER ${CMAKE_C_COMPILER})

# 5. Configure how CMake finds libraries and headers. This forces it to
#    look only in our specified sysroot and not elsewhere.
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH ${CMAKE_SYSROOT})

# 6. Add necessary compile flags for Android libraries.
add_compile_options(-fPIC)
add_compile_options(--sysroot=${CMAKE_SYSROOT})