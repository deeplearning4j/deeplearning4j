# Android ARM64 toolchain with binary execution testing
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

# Set NDK paths
set(ANDROID_NDK_ROOT $ENV{ANDROID_NDK_ROOT})
set(TOOLCHAIN_DIR "${ANDROID_NDK_ROOT}/toolchains/llvm/prebuilt/linux-aarch64")
set(CMAKE_SYSROOT "${TOOLCHAIN_DIR}/sysroot")

message(STATUS "=== BINARY EXECUTION TESTING ===")

# Test each binary candidate thoroughly
set(TEST_BINARIES
        "${TOOLCHAIN_DIR}/bin/clang"
        "${TOOLCHAIN_DIR}/bin/clang++"
        "${TOOLCHAIN_DIR}/bin/aarch64-linux-android21-clang"
        "${TOOLCHAIN_DIR}/bin/aarch64-linux-android21-clang++"
)

foreach(BINARY ${TEST_BINARIES})
   if(EXISTS ${BINARY})
      get_filename_component(BINARY_NAME ${BINARY} NAME)
      message(STATUS "Testing ${BINARY_NAME}...")

      # Test 1: Check file type
      execute_process(
              COMMAND file ${BINARY}
              OUTPUT_VARIABLE FILE_TYPE
              OUTPUT_STRIP_TRAILING_WHITESPACE
              ERROR_QUIET
      )
      message(STATUS "  File type: ${FILE_TYPE}")

      # Test 2: Check dependencies
      execute_process(
              COMMAND ldd ${BINARY}
              OUTPUT_VARIABLE LDD_OUTPUT
              OUTPUT_STRIP_TRAILING_WHITESPACE
              ERROR_VARIABLE LDD_ERROR
              RESULT_VARIABLE LDD_RESULT
      )
      if(LDD_RESULT EQUAL 0)
         message(STATUS "  Dependencies: ${LDD_OUTPUT}")
      else()
         message(STATUS "  Dependencies error: ${LDD_ERROR}")
      endif()

      # Test 3: Try to execute with --version
      execute_process(
              COMMAND ${BINARY} --version
              OUTPUT_VARIABLE VERSION_OUTPUT
              ERROR_VARIABLE VERSION_ERROR
              RESULT_VARIABLE VERSION_RESULT
              TIMEOUT 10
      )

      if(VERSION_RESULT EQUAL 0)
         string(SUBSTRING "${VERSION_OUTPUT}" 0 100 VERSION_BRIEF)
         message(STATUS "  ✓ WORKS: ${VERSION_BRIEF}")
         if(NOT CMAKE_C_COMPILER AND BINARY_NAME MATCHES "clang$")
            set(CMAKE_C_COMPILER ${BINARY})
            message(STATUS "  Selected as C compiler")
         endif()
         if(NOT CMAKE_CXX_COMPILER AND BINARY_NAME MATCHES "clang\\+\\+$")
            set(CMAKE_CXX_COMPILER ${BINARY})
            message(STATUS "  Selected as C++ compiler")
         endif()
      else()
         message(STATUS "  ✗ FAILED (exit code ${VERSION_RESULT}): ${VERSION_ERROR}")
      endif()

      message(STATUS "  ---")
   endif()
endforeach()

# If no C++ compiler found, use C compiler
if(NOT CMAKE_CXX_COMPILER AND CMAKE_C_COMPILER)
   set(CMAKE_CXX_COMPILER ${CMAKE_C_COMPILER})
   message(STATUS "Using C compiler for C++: ${CMAKE_CXX_COMPILER}")
endif()

# Verify we have working compilers
if(NOT CMAKE_C_COMPILER)
   message(FATAL_ERROR "No working C compiler found")
endif()

if(NOT CMAKE_CXX_COMPILER)
   message(FATAL_ERROR "No working C++ compiler found")
endif()

set(CMAKE_ASM_COMPILER ${CMAKE_C_COMPILER})

# Set compiler flags
set(CMAKE_C_FLAGS "--target=aarch64-linux-android21 --sysroot=${CMAKE_SYSROOT}")
set(CMAKE_CXX_FLAGS "--target=aarch64-linux-android21 --sysroot=${CMAKE_SYSROOT} -stdlib=libc++")
set(CMAKE_ASM_FLAGS "--target=aarch64-linux-android21 --sysroot=${CMAKE_SYSROOT}")

# Cross-compilation settings
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH ${CMAKE_SYSROOT})

message(STATUS "=== FINAL SELECTION ===")
message(STATUS "C compiler: ${CMAKE_C_COMPILER}")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "ASM compiler: ${CMAKE_ASM_COMPILER}")
message(STATUS "========================")