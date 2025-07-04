# android-arm64.cmake - Manual Toolchain for non-standard ARM64-hosted NDK
#
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
#    This assumes a 'linux-aarch64' host toolchain, common for ARM64 Linux systems.
set(TOOLCHAIN_DIR "${ANDROID_NDK_ROOT}/toolchains/llvm/prebuilt/linux-aarch64")
set(CMAKE_SYSROOT "${TOOLCHAIN_DIR}/sysroot")

# 4. Explicitly set the C/C++ compilers to the *main binaries*, not the broken scripts.
set(CMAKE_C_COMPILER "${TOOLCHAIN_DIR}/bin/clang")
set(CMAKE_CXX_COMPILER "${TOOLCHAIN_DIR}/bin/clang++")
set(CMAKE_ASM_COMPILER ${CMAKE_C_COMPILER})

# 5. Configure how CMake finds libraries and headers within the defined sysroot.
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH ${CMAKE_SYSROOT})

# 6. Manually set the flags the broken script was supposed to add.
#    Setting CMAKE_<LANG>_FLAGS directly ensures they are used by CMake's
#    initial compiler check, preventing the "compiler is not able to compile
#    a simple test program" error.
set(ANDROID_COMPILE_FLAGS
        "--target=aarch64-linux-android21"
        "--sysroot=${CMAKE_SYSROOT}"
        -fPIC
)

set(CMAKE_C_FLAGS "${ANDROID_COMPILE_FLAGS}" CACHE STRING "C compiler flags")
set(CMAKE_CXX_FLAGS "${ANDROID_COMPILE_FLAGS}" CACHE STRING "C++ compiler flags")
set(CMAKE_ASM_FLAGS "${ANDROID_COMPILE_FLAGS}" CACHE STRING "Assembler flags")