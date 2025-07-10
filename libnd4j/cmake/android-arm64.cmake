# android-arm64.cmake - Flexible toolchain for NDK r27b+ and cross-platform builds

# Set the system and processor
set(CMAKE_SYSTEM_NAME Linux)

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

# Use unified headers (available since NDK r14)
set(CMAKE_ANDROID_STL_TYPE c++_shared)

# Set toolchain paths with flexibility for different NDK structures
set(NDK_TOOLCHAIN_PATH "${CMAKE_ANDROID_NDK}/toolchains/llvm/prebuilt/linux-x86_64")

# Check if the toolchain path exists
if(NOT EXISTS "${NDK_TOOLCHAIN_PATH}")
   message(FATAL_ERROR "NDK toolchain path does not exist: ${NDK_TOOLCHAIN_PATH}")
endif()

# Fix for NDK r27b+ structure - use unified sysroot
set(CMAKE_SYSROOT "${NDK_TOOLCHAIN_PATH}/sysroot")

# Verify sysroot exists
if(NOT EXISTS "${CMAKE_SYSROOT}")
   message(FATAL_ERROR "NDK sysroot does not exist: ${CMAKE_SYSROOT}")
endif()

# Use system compilers instead of NDK compilers to avoid GLIBC issues
set(CMAKE_C_COMPILER "/usr/bin/clang")
set(CMAKE_CXX_COMPILER "/usr/bin/clang++")

# Set target and API level flags for cross compilation with NDK sysroot and libc
set(ANDROID_TARGET_FLAGS "-target aarch64-linux-android${ANDROID_NATIVE_API_LEVEL} --sysroot=${CMAKE_SYSROOT}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${ANDROID_TARGET_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ANDROID_TARGET_FLAGS}")

# Ensure we use NDK's libc and not system libc
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --sysroot=${CMAKE_SYSROOT}")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} --sysroot=${CMAKE_SYSROOT}")

message(STATUS "Using system compilers with Android target flags and NDK sysroot")
message(STATUS "Final C compiler: ${CMAKE_C_COMPILER}")
message(STATUS "Final C++ compiler: ${CMAKE_CXX_COMPILER}")

# Set the find root path
set(CMAKE_FIND_ROOT_PATH "${CMAKE_SYSROOT}")
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# Architecture-specific flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=armv8-a -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8-a -fPIC")

# Additional Android-specific flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -ffunction-sections -fdata-sections")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffunction-sections -fdata-sections")

# Linker flags
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--gc-sections -Wl,--as-needed")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--gc-sections -Wl,--as-needed")

# Debug information
message(STATUS "Android NDK: ${CMAKE_ANDROID_NDK}")
message(STATUS "Android API Level: ${ANDROID_NATIVE_API_LEVEL}")
message(STATUS "Android ABI: ${CMAKE_ANDROID_ARCH_ABI}")
message(STATUS "Android Sysroot: ${CMAKE_SYSROOT}")
message(STATUS "C Compiler: ${CMAKE_C_COMPILER}")
message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER}")

# Compatibility layer for older CMake/NDK combinations
# Create the expected legacy directory structure if it doesn't exist
set(LEGACY_PLATFORM_DIR "${CMAKE_ANDROID_NDK}/platforms/android-${ANDROID_NATIVE_API_LEVEL}/arch-arm64")
if(NOT EXISTS "${LEGACY_PLATFORM_DIR}")
   message(STATUS "Creating legacy platform directory structure for compatibility...")
   file(MAKE_DIRECTORY "${LEGACY_PLATFORM_DIR}")

   # Create symbolic links if on Unix-like system
   if(UNIX)
      execute_process(
              COMMAND ${CMAKE_COMMAND} -E create_symlink
              "${CMAKE_SYSROOT}/usr"
              "${LEGACY_PLATFORM_DIR}/usr"
              ERROR_QUIET
      )
   endif()
endif()
