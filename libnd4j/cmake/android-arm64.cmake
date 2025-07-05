# android-arm64.cmake - CMake toolchain for Android ARM64 cross-compilation
# Designed to work with non-standard ARM64-hosted NDK (Termux NDK)

# Set target system
set(CMAKE_SYSTEM_NAME Android)
set(CMAKE_SYSTEM_VERSION 21)
set(CMAKE_SYSTEM_PROCESSOR aarch64)
set(CMAKE_ANDROID_ARCH_ABI arm64-v8a)
set(CMAKE_ANDROID_NDK ${ANDROID_NDK_ROOT})

# Get NDK root from environment
set(ANDROID_NDK_ROOT $ENV{ANDROID_NDK_ROOT})
if(NOT ANDROID_NDK_ROOT)
   message(FATAL_ERROR "ANDROID_NDK_ROOT environment variable is not set.")
endif()

# Set paths
set(TOOLCHAIN_DIR "${ANDROID_NDK_ROOT}/toolchains/llvm/prebuilt/linux-aarch64")
set(CMAKE_SYSROOT "${TOOLCHAIN_DIR}/sysroot")

# Debug output
message(STATUS "=== Android ARM64 Toolchain Debug ===")
message(STATUS "ANDROID_NDK_ROOT: ${ANDROID_NDK_ROOT}")
message(STATUS "TOOLCHAIN_DIR: ${TOOLCHAIN_DIR}")
message(STATUS "CMAKE_SYSROOT: ${CMAKE_SYSROOT}")
message(STATUS "Environment CMAKE_C_COMPILER: '$ENV{CMAKE_C_COMPILER}'")
message(STATUS "Environment CMAKE_CXX_COMPILER: '$ENV{CMAKE_CXX_COMPILER}'")

# Check if toolchain directory exists
if(NOT EXISTS ${TOOLCHAIN_DIR})
   message(FATAL_ERROR "Toolchain directory does not exist: ${TOOLCHAIN_DIR}")
endif()

# Find working C compiler
set(CMAKE_C_COMPILER "")
if(DEFINED ENV{CMAKE_C_COMPILER} AND NOT "$ENV{CMAKE_C_COMPILER}" STREQUAL "")
   if(EXISTS "$ENV{CMAKE_C_COMPILER}")
      set(CMAKE_C_COMPILER $ENV{CMAKE_C_COMPILER})
      message(STATUS "Using C compiler from environment: ${CMAKE_C_COMPILER}")
   else()
      message(STATUS "Environment C compiler does not exist: $ENV{CMAKE_C_COMPILER}")
   endif()
endif()

# If no compiler from environment, search for available ones
if(CMAKE_C_COMPILER STREQUAL "")
   # List of possible C compiler locations in order of preference
   set(C_COMPILER_CANDIDATES
           "${TOOLCHAIN_DIR}/bin/android-clang-wrapper"
           "${TOOLCHAIN_DIR}/bin/aarch64-linux-android21-clang"
           "${TOOLCHAIN_DIR}/bin/clang"
           "${TOOLCHAIN_DIR}/bin/clang-18"
   )

   foreach(CANDIDATE ${C_COMPILER_CANDIDATES})
      if(EXISTS ${CANDIDATE})
         set(CMAKE_C_COMPILER ${CANDIDATE})
         message(STATUS "Found C compiler: ${CMAKE_C_COMPILER}")
         break()
      endif()
   endforeach()
endif()

# Verify we found a C compiler
if(CMAKE_C_COMPILER STREQUAL "")
   message(STATUS "Listing contents of ${TOOLCHAIN_DIR}/bin/:")
   file(GLOB BIN_CONTENTS "${TOOLCHAIN_DIR}/bin/*")
   foreach(ITEM ${BIN_CONTENTS})
      message(STATUS "  ${ITEM}")
   endforeach()
   message(FATAL_ERROR "Could not find any working C compiler")
endif()

# Find working C++ compiler
set(CMAKE_CXX_COMPILER "")
if(DEFINED ENV{CMAKE_CXX_COMPILER} AND NOT "$ENV{CMAKE_CXX_COMPILER}" STREQUAL "")
   if(EXISTS "$ENV{CMAKE_CXX_COMPILER}")
      set(CMAKE_CXX_COMPILER $ENV{CMAKE_CXX_COMPILER})
      message(STATUS "Using C++ compiler from environment: ${CMAKE_CXX_COMPILER}")
   else()
      message(STATUS "Environment C++ compiler does not exist: $ENV{CMAKE_CXX_COMPILER}")
   endif()
endif()

# If no C++ compiler from environment, search for available ones or use C compiler
if(CMAKE_CXX_COMPILER STREQUAL "")
   # List of possible C++ compiler locations in order of preference
   set(CXX_COMPILER_CANDIDATES
           "${TOOLCHAIN_DIR}/bin/android-clang++-wrapper"
           "${TOOLCHAIN_DIR}/bin/aarch64-linux-android21-clang++"
           "${TOOLCHAIN_DIR}/bin/clang++"
           "${CMAKE_C_COMPILER}"
   )

   foreach(CANDIDATE ${CXX_COMPILER_CANDIDATES})
      if(EXISTS ${CANDIDATE})
         set(CMAKE_CXX_COMPILER ${CANDIDATE})
         message(STATUS "Found C++ compiler: ${CMAKE_CXX_COMPILER}")
         break()
      endif()
   endforeach()
endif()

# Verify we found a C++ compiler
if(CMAKE_CXX_COMPILER STREQUAL "")
   message(FATAL_ERROR "Could not find any working C++ compiler")
endif()

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

# Final debug output
message(STATUS "=== Final Compiler Configuration ===")
message(STATUS "CMAKE_C_COMPILER: ${CMAKE_C_COMPILER}")
message(STATUS "CMAKE_CXX_COMPILER: ${CMAKE_CXX_COMPILER}")
message(STATUS "CMAKE_ASM_COMPILER: ${CMAKE_ASM_COMPILER}")
message(STATUS "CMAKE_C_FLAGS: ${CMAKE_C_FLAGS}")
message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
message(STATUS "CMAKE_SYSTEM_VERSION: ${CMAKE_SYSTEM_VERSION}")
message(STATUS "CMAKE_ANDROID_ARCH_ABI: ${CMAKE_ANDROID_ARCH_ABI}")
message(STATUS "=======================================")