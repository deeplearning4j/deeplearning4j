# Android ARM64 toolchain using system clang with Android sysroot
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

# Set NDK paths for sysroot
set(ANDROID_NDK_ROOT $ENV{ANDROID_NDK_ROOT})
set(TOOLCHAIN_DIR "${ANDROID_NDK_ROOT}/toolchains/llvm/prebuilt/linux-aarch64")
set(CMAKE_SYSROOT "${TOOLCHAIN_DIR}/sysroot")

message(STATUS "=== SEARCHING FOR SYSTEM CLANG ===")

# Search for working clang binaries on the system
set(SYSTEM_CLANG_CANDIDATES
        "/usr/bin/clang-18"
        "/usr/bin/clang-17"
        "/usr/bin/clang-16"
        "/usr/bin/clang-15"
        "/usr/bin/clang"
        "/usr/local/bin/clang"
        "/opt/llvm/bin/clang"
)

set(WORKING_CLANG "")

foreach(CLANG_PATH ${SYSTEM_CLANG_CANDIDATES})
   if(EXISTS ${CLANG_PATH})
      message(STATUS "Testing system clang: ${CLANG_PATH}")

      # Test if it can cross-compile for Android ARM64
      execute_process(
              COMMAND ${CLANG_PATH} --target=aarch64-linux-android21 --version
              OUTPUT_VARIABLE VERSION_OUTPUT
              ERROR_VARIABLE VERSION_ERROR
              RESULT_VARIABLE VERSION_RESULT
              TIMEOUT 10
      )

      if(VERSION_RESULT EQUAL 0)
         # Test if it can find the Android sysroot
         execute_process(
                 COMMAND ${CLANG_PATH} --target=aarch64-linux-android21 --sysroot=${CMAKE_SYSROOT} -E -v -x c /dev/null
                 OUTPUT_VARIABLE SYSROOT_TEST
                 ERROR_VARIABLE SYSROOT_ERROR
                 RESULT_VARIABLE SYSROOT_RESULT
                 TIMEOUT 10
         )

         if(SYSROOT_RESULT EQUAL 0)
            string(SUBSTRING "${VERSION_OUTPUT}" 0 100 VERSION_BRIEF)
            message(STATUS "  ✓ WORKS: ${VERSION_BRIEF}")
            set(WORKING_CLANG ${CLANG_PATH})
            break()
         else()
            message(STATUS "  ✗ Cannot use Android sysroot: ${SYSROOT_ERROR}")
         endif()
      else()
         message(STATUS "  ✗ Cannot target Android ARM64: ${VERSION_ERROR}")
      endif()
   endif()
endforeach()

if(NOT WORKING_CLANG)
   message(STATUS "No system clang found, trying to find any clang...")

   # Last resort: find ANY clang binary
   find_program(FOUND_CLANG NAMES clang-18 clang-17 clang-16 clang-15 clang)

   if(FOUND_CLANG)
      message(STATUS "Found clang at: ${FOUND_CLANG}")
      execute_process(
              COMMAND ${FOUND_CLANG} --target=aarch64-linux-android21 --version
              RESULT_VARIABLE TEST_RESULT
              OUTPUT_QUIET ERROR_QUIET
      )

      if(TEST_RESULT EQUAL 0)
         set(WORKING_CLANG ${FOUND_CLANG})
         message(STATUS "Using found clang: ${WORKING_CLANG}")
      endif()
   endif()
endif()

if(NOT WORKING_CLANG)
   message(FATAL_ERROR "No working clang compiler found that can target Android ARM64")
endif()

# Use the working system clang
set(CMAKE_C_COMPILER ${WORKING_CLANG})
set(CMAKE_CXX_COMPILER ${WORKING_CLANG})
set(CMAKE_ASM_COMPILER ${WORKING_CLANG})

# Set comprehensive Android cross-compilation flags
set(CMAKE_C_FLAGS "--target=aarch64-linux-android21 --sysroot=${CMAKE_SYSROOT} -fPIC")
set(CMAKE_CXX_FLAGS "--target=aarch64-linux-android21 --sysroot=${CMAKE_SYSROOT} -fPIC -stdlib=libc++")
set(CMAKE_ASM_FLAGS "--target=aarch64-linux-android21 --sysroot=${CMAKE_SYSROOT}")

# Set linker flags
set(CMAKE_EXE_LINKER_FLAGS "--target=aarch64-linux-android21 --sysroot=${CMAKE_SYSROOT}")
set(CMAKE_SHARED_LINKER_FLAGS "--target=aarch64-linux-android21 --sysroot=${CMAKE_SYSROOT}")

# Cross-compilation settings
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH ${CMAKE_SYSROOT})

# Final verification
message(STATUS "=== FINAL VERIFICATION ===")
execute_process(
        COMMAND ${CMAKE_C_COMPILER} ${CMAKE_C_FLAGS} -E -x c /dev/null
        OUTPUT_VARIABLE FINAL_TEST
        ERROR_VARIABLE FINAL_ERROR
        RESULT_VARIABLE FINAL_RESULT
        TIMEOUT 15
)

if(FINAL_RESULT EQUAL 0)
   message(STATUS "✓ Final cross-compilation test PASSED")
else()
   message(WARNING "✗ Final test failed but proceeding anyway: ${FINAL_ERROR}")
endif()

message(STATUS "=== FINAL CONFIGURATION ===")
message(STATUS "C compiler: ${CMAKE_C_COMPILER}")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "Sysroot: ${CMAKE_SYSROOT}")
message(STATUS "C flags: ${CMAKE_C_FLAGS}")
message(STATUS "C++ flags: ${CMAKE_CXX_FLAGS}")
message(STATUS "===============================")