# Diagnostic toolchain to find what actually exists in the NDK
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

# Set NDK paths
set(ANDROID_NDK_ROOT $ENV{ANDROID_NDK_ROOT})
message(STATUS "=== NDK DIAGNOSTIC ===")
message(STATUS "ANDROID_NDK_ROOT: ${ANDROID_NDK_ROOT}")

if(NOT ANDROID_NDK_ROOT)
   message(FATAL_ERROR "ANDROID_NDK_ROOT not set")
endif()

# Check if basic paths exist
set(TOOLCHAIN_DIR "${ANDROID_NDK_ROOT}/toolchains/llvm/prebuilt/linux-aarch64")
message(STATUS "Expected toolchain dir: ${TOOLCHAIN_DIR}")

if(EXISTS ${TOOLCHAIN_DIR})
   message(STATUS "✓ Toolchain directory exists")

   # List contents of bin directory
   set(BIN_DIR "${TOOLCHAIN_DIR}/bin")
   if(EXISTS ${BIN_DIR})
      message(STATUS "✓ Bin directory exists: ${BIN_DIR}")
      file(GLOB BIN_FILES "${BIN_DIR}/*")
      message(STATUS "=== BIN DIRECTORY CONTENTS ===")
      foreach(FILE ${BIN_FILES})
         get_filename_component(FILENAME ${FILE} NAME)
         if(IS_DIRECTORY ${FILE})
            message(STATUS "  [DIR]  ${FILENAME}")
         else()
            # Check if executable
            execute_process(
                    COMMAND test -x ${FILE}
                    RESULT_VARIABLE IS_EXECUTABLE
                    OUTPUT_QUIET ERROR_QUIET
            )
            if(IS_EXECUTABLE EQUAL 0)
               message(STATUS "  [EXEC] ${FILENAME}")
            else()
               message(STATUS "  [FILE] ${FILENAME}")
            endif()
         endif()
      endforeach()
   else()
      message(STATUS "✗ Bin directory does not exist")
   endif()

   # Check sysroot
   set(SYSROOT_DIR "${TOOLCHAIN_DIR}/sysroot")
   if(EXISTS ${SYSROOT_DIR})
      message(STATUS "✓ Sysroot exists: ${SYSROOT_DIR}")
   else()
      message(STATUS "✗ Sysroot does not exist")
   endif()

else()
   message(STATUS "✗ Toolchain directory does not exist")

   # Try alternative paths
   message(STATUS "=== SEARCHING FOR ALTERNATIVE PATHS ===")

   # Check different host architectures
   set(ALT_PATHS
           "${ANDROID_NDK_ROOT}/toolchains/llvm/prebuilt/linux-x86_64"
           "${ANDROID_NDK_ROOT}/toolchains/llvm/prebuilt/darwin-x86_64"
           "${ANDROID_NDK_ROOT}/bin"
           "${ANDROID_NDK_ROOT}/toolchain/bin"
   )

   foreach(ALT_PATH ${ALT_PATHS})
      if(EXISTS ${ALT_PATH})
         message(STATUS "✓ Found alternative: ${ALT_PATH}")
         if(EXISTS "${ALT_PATH}/bin")
            file(GLOB ALT_FILES "${ALT_PATH}/bin/*clang*")
            foreach(FILE ${ALT_FILES})
               get_filename_component(FILENAME ${FILE} NAME)
               message(STATUS "    ${FILENAME}")
            endforeach()
         endif()
      else()
         message(STATUS "✗ Not found: ${ALT_PATH}")
      endif()
   endforeach()
endif()

# Try to find ANY clang binary anywhere in the NDK
message(STATUS "=== SEARCHING FOR ANY CLANG BINARY ===")
execute_process(
        COMMAND find ${ANDROID_NDK_ROOT} -name "*clang*" -type f -executable
        OUTPUT_VARIABLE FOUND_CLANG_FILES
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
)

if(FOUND_CLANG_FILES)
   string(REPLACE "\n" ";" CLANG_LIST ${FOUND_CLANG_FILES})
   foreach(CLANG_FILE ${CLANG_LIST})
      message(STATUS "Found clang: ${CLANG_FILE}")
   endforeach()
else()
   message(STATUS "No clang binaries found anywhere in NDK")
endif()

message(STATUS "=== END DIAGNOSTIC ===")

# Fail the configuration so we can see the output
message(FATAL_ERROR "Diagnostic complete - check output above")