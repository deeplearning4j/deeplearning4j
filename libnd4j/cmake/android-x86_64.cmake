# Alternative approach: Don't use CMAKE_SYSTEM_NAME Android
# This bypasses CMake's built-in Android support that's causing the override

# Set the system as Linux instead of Android
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR x86_64)  # Changed from aarch64

# Set Android-specific variables manually
set(ANDROID TRUE)
set(ANDROID_NATIVE_API_LEVEL 21)
set(CMAKE_ANDROID_ARCH_ABI x86_64)  # Changed from arm64-v8a

# NDK path detection (same as before)
if(NOT DEFINED CMAKE_ANDROID_NDK)
   if(DEFINED ENV{ANDROID_NDK})
      set(CMAKE_ANDROID_NDK $ENV{ANDROID_NDK})
   else()
      message(FATAL_ERROR "Android NDK not found. Please set ANDROID_NDK environment variable")
   endif()
endif()

# Host detection (same as before)
if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
   set(NDK_HOST_TAG "linux-aarch64")
else()
   set(NDK_HOST_TAG "linux-x86_64")
endif()

# Set compilers directly without CMake Android interference
set(NDK_TOOLCHAIN_PATH "${CMAKE_ANDROID_NDK}/toolchains/llvm/prebuilt/${NDK_HOST_TAG}")
set(CMAKE_C_COMPILER "${NDK_TOOLCHAIN_PATH}/bin/x86_64-linux-android21-clang")      # Changed from aarch64
set(CMAKE_CXX_COM