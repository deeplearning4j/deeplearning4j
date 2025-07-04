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

# 4. Use the compilers from environment variables set by the workflow,
#    or find the actual working binaries.
if(DEFINED ENV{CMAKE_C_COMPILER})
   set(CMAKE_C_COMPILER $ENV{CMAKE_C_COMPILER})
   message(STATUS "Using C compiler from environment: ${CMAKE_C_COMPILER}")
else()
   # Try to find the actual clang binary that exists
   set(CLANG_CANDIDATES
           "${TOOLCHAIN_DIR}/bin/aarch64-linux-android21-clang"
           "${TOOLCHAIN_DIR}/bin/clang"
           "${TOOLCHAIN_DIR}/bin/clang-18"
   )

   foreach(CANDIDATE ${CLANG_CANDIDATES})
      if(EXISTS ${CANDIDATE})
         set(CMAKE_C_COMPILER ${CANDIDATE})
         message(STATUS "Found C compiler: ${CMAKE_C_COMPILER}")
         break()
      endif()
   endforeach()

   if(NOT CMAKE_C_COMPILER)
      message(FATAL_ERROR "Could not find clang compiler in ${TOOLCHAIN_DIR}/bin/")
   endif()
endif()

if(DEFINED ENV{CMAKE_CXX_COMPILER})
   set(CMAKE_CXX_COMPILER $ENV{CMAKE_CXX_COMPILER})
   message(STATUS "Using C++ compiler from environment: ${CMAKE_CXX_COMPILER}")
else()
   # Try to find the actual clang++ binary that exists
   set(CLANGXX_CANDIDATES
           "${TOOLCHAIN_DIR}/bin/aarch64-linux-android21-clang++"
           "${TOOLCHAIN_DIR}/bin/clang++"
           "${CMAKE_C_COMPILER}++"
           "${CMAKE_C_COMPILER}"
   )

   foreach(CANDIDATE ${CLANGXX_CANDIDATES})
      if(EXISTS ${CANDIDATE})
         set(CMAKE_CXX_COMPILER ${CANDIDATE})
         message(STATUS "Found C++ compiler: ${CMAKE_CXX_COMPILER}")
         break()
      endif()
   endforeach()

   if(NOT CMAKE_CXX_COMPILER)
      message(FATAL_ERROR "Could not find clang++ compiler in ${TOOLCHAIN_DIR}/bin/")
   endif()
endif()

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
set(CMAKE_C_FLAGS "--target=aarch64-linux-android21 --sysroot=${CMAKE_SYSROOT}" CACHE STRING "C compiler flags")
set(CMAKE_CXX_FLAGS "--target=aarch64-linux-android21 --sysroot=${CMAKE_SYSROOT}" CACHE STRING "C++ compiler flags")
set(CMAKE_ASM_FLAGS "--target=aarch64-linux-android21 --sysroot=${CMAKE_SYSROOT}" CACHE STRING "Assembler flags")