# Android ARM64 toolchain that uses direct binaries and bypasses broken symlinks
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

# Set NDK paths
set(ANDROID_NDK_ROOT $ENV{ANDROID_NDK_ROOT})
set(TOOLCHAIN_DIR "${ANDROID_NDK_ROOT}/toolchains/llvm/prebuilt/linux-aarch64")
set(CMAKE_SYSROOT "${TOOLCHAIN_DIR}/sysroot")

message(STATUS "=== FIXING BROKEN SYMLINKS ===")

# Find working direct binaries (not symlinks)
set(DIRECT_BINARIES
        "${TOOLCHAIN_DIR}/bin/clang-18"
        "${TOOLCHAIN_DIR}/bin/clang-17"
)

set(WORKING_CLANG "")

foreach(BINARY ${DIRECT_BINARIES})
   if(EXISTS ${BINARY})
      get_filename_component(BINARY_NAME ${BINARY} NAME)
      message(STATUS "Testing direct binary ${BINARY_NAME}...")

      # Test execution
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
         set(WORKING_CLANG ${BINARY})
         break()
      else()
         message(STATUS "  ✗ FAILED: ${VERSION_ERROR}")
      endif()
   endif()
endforeach()

if(NOT WORKING_CLANG)
   message(FATAL_ERROR "No working direct clang binary found")
endif()

# Use the working binary directly
set(CMAKE_C_COMPILER ${WORKING_CLANG})
set(CMAKE_CXX_COMPILER ${WORKING_CLANG})  # Use same binary for C++
set(CMAKE_ASM_COMPILER ${WORKING_CLANG})

message(STATUS "Using working binary: ${WORKING_CLANG}")

# Set compiler flags with proper Android targeting
set(CMAKE_C_FLAGS "--target=aarch64-linux-android21 --sysroot=${CMAKE_SYSROOT}")
set(CMAKE_CXX_FLAGS "--target=aarch64-linux-android21 --sysroot=${CMAKE_SYSROOT} -stdlib=libc++")
set(CMAKE_ASM_FLAGS "--target=aarch64-linux-android21 --sysroot=${CMAKE_SYSROOT}")

# Cross-compilation settings
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH ${CMAKE_SYSROOT})

# Test final compiler one more time
message(STATUS "=== FINAL COMPILER TEST ===")
execute_process(
        COMMAND ${CMAKE_C_COMPILER} --target=aarch64-linux-android21 --sysroot=${CMAKE_SYSROOT} --version
        OUTPUT_VARIABLE FINAL_TEST
        ERROR_VARIABLE FINAL_ERROR
        RESULT_VARIABLE FINAL_RESULT
        TIMEOUT 10
)

if(FINAL_RESULT EQUAL 0)
   string(SUBSTRING "${FINAL_TEST}" 0 200 FINAL_BRIEF)
   message(STATUS "✓ Final compiler test PASSED: ${FINAL_BRIEF}")
else()
   message(FATAL_ERROR "✗ Final compiler test FAILED: ${FINAL_ERROR}")
endif()

message(STATUS "=== COMPILER CONFIGURATION ===")
message(STATUS "C compiler: ${CMAKE_C_COMPILER}")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "ASM compiler: ${CMAKE_ASM_COMPILER}")
message(STATUS "C flags: ${CMAKE_C_FLAGS}")
message(STATUS "C++ flags: ${CMAKE_CXX_FLAGS}")
message(STATUS "===============================")