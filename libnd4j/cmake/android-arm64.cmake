# android-arm64.cmake - Manual Toolchain for non-standard ARM64-hosted NDK

# This version bypasses the NDK's broken wrapper scripts by calling the main
# clang/clang++ binaries directly and setting the --target flag manually.

# 1. Set the target system processor.
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

# 2. Point to the NDK root from the environment variable.
set(ANDROID_NDK_ROOT $ENV{ANDROID_NDK_ROOT})
if(NOT ANDROID_NDK_ROOT)
   message(FATAL_ERROR "ANDROID_NDK_ROOT environment variable is not set.")
endif()

# 3. Manually define the paths to the toolchain directory and the system root.
set(TOOLCHAIN_DIR "${ANDROID_NDK_ROOT}/toolchains/llvm/prebuilt/linux-aarch64")
set(CMAKE_SYSROOT "${TOOLCHAIN_DIR}/sysroot")

# 4. Explicitly set the C/C++ compilers to the *main binaries*, not the broken scripts.
set(CMAKE_C_COMPILER "${TOOLCHAIN_DIR}/bin/clang")
set(CMAKE_CXX_COMPILER "${TOOLCHAIN_DIR}/bin/clang++")
set(CMAKE_ASM_COMPILER ${CMAKE_C_COMPILER})

# 5. Configure how CMake finds libraries and headers.
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH ${CMAKE_SYSROOT})

# 6. Manually add the flags the broken script was supposed to add.
#    This is the key to making the main clang binary work for cross-compilation.
add_compile_options(-fPIC)
add_compile_options(--sysroot=${CMAKE_SYSROOT})
add_compile_options(--target=aarch64-linux-android21)