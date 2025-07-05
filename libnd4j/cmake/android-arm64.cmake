# Minimal Android ARM64 toolchain - bypass all CMake detection
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

# Set NDK paths
set(ANDROID_NDK_ROOT $ENV{ANDROID_NDK_ROOT})
set(TOOLCHAIN_DIR "${ANDROID_NDK_ROOT}/toolchains/llvm/prebuilt/linux-aarch64")
set(CMAKE_SYSROOT "${TOOLCHAIN_DIR}/sysroot")

# Force specific compiler paths - use clang directly, avoid all wrapper scripts
set(CMAKE_C_COMPILER "${TOOLCHAIN_DIR}/bin/clang")
set(CMAKE_CXX_COMPILER "${TOOLCHAIN_DIR}/bin/clang")  # Use clang for C++ too
set(CMAKE_ASM_COMPILER "${TOOLCHAIN_DIR}/bin/clang")

# Disable compiler checks - assume they work
set(CMAKE_C_COMPILER_FORCED TRUE)
set(CMAKE_CXX_COMPILER_FORCED TRUE)
set(CMAKE_ASM_COMPILER_FORCED TRUE)

# Set flags
set(CMAKE_C_FLAGS "--target=aarch64-linux-android21 --sysroot=${CMAKE_SYSROOT}")
set(CMAKE_CXX_FLAGS "--target=aarch64-linux-android21 --sysroot=${CMAKE_SYSROOT} -stdlib=libc++")
set(CMAKE_ASM_FLAGS "--target=aarch64-linux-android21 --sysroot=${CMAKE_SYSROOT}")

# Cross-compilation settings
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH ${CMAKE_SYSROOT})